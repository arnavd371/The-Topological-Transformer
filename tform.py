"""
Topological Transformer (T-Form)
=================================
Act as a Senior Research Engineer specializing in Geometric Deep Learning.

Architecture
------------
Input: A sequence of latent embeddings (B, L, D).

TDA Layer:
    For each sequence, treat the L tokens as a point cloud in D-dimensional
    space. Use the Gudhi library to compute a Vietoris-Rips filtration and
    extract 0-dimensional (connectivity) and 1-dimensional (loop) persistence
    features.

Topological Encoding:
    Map the persistence diagrams (birth/death pairs) into a fixed-size
    "Persistence Image" vector via weighted Gaussian density estimation on the
    birth/persistence plane.

Topo-Attention:
    Use the topological signature as an additive bias in the scaled
    dot-product attention matrix so the model attends to "structural holes"
    in the logic:

        Attention(Q, K, V) = Softmax( (Q K^T / sqrt(d_k)) + Φ(PD) ) V

    where Φ(PD) is a neural network that processes the Persistence Diagram.

Saddle Point Features:
    Tokens that connect previously disjoint components (0-cycles) at the
    moment of their death are identified as "saddle" tokens.  Their indices
    are stored and injected as learnable bias offsets into the attention
    logits so the model can learn to emphasise logical bridging tokens.

Differentiability Bridge (Frozen Filtration):
    Gudhi's Vietoris-Rips computations are non-differentiable.  We adopt the
    "Persistence Image Path": all TDA calls are wrapped in torch.no_grad()
    and the resulting fixed-size persistence image is then processed by a
    small trainable MLP (TopologicalBiasNet) whose gradients flow normally.
    This keeps the full PyTorch autograd graph intact for the learnable parts
    of the model.

Usage
-----
    >>> import torch
    >>> from tform import TopologicalTransformer
    >>> model = TopologicalTransformer(d_model=64, num_heads=4, num_layers=2,
    ...                                max_seq_len=16)
    >>> x = torch.randn(2, 16, 64)           # (batch, seq_len, d_model)
    >>> out = model(x)                        # (batch, seq_len, d_model)
    >>> out.shape
    torch.Size([2, 16, 64])
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gudhi  # type: ignore
    _GUDHI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _GUDHI_AVAILABLE = False
    warnings.warn(
        "gudhi is not installed.  Install it with: pip install gudhi\n"
        "Topological features will be replaced by zero vectors.",
        ImportWarning,
        stacklevel=2,
    )

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_IMG_RESOLUTION = 10    # persistence image grid size per axis (10×10 = 100)
_MAX_FILTRATION = 10.0  # clip filtration values to this radius


# ---------------------------------------------------------------------------
# 1. Persistence Diagram Computation (non-differentiable, frozen)
# ---------------------------------------------------------------------------

def _compute_persistence_diagrams(
    points: np.ndarray,
    max_edge_length: float = _MAX_FILTRATION,
    max_dimension: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a Vietoris-Rips filtration on *points* and return raw
    persistence diagrams for dimensions 0 and 1.

    Parameters
    ----------
    points:
        Array of shape ``(L, D)`` — the token embeddings treated as a point
        cloud in D-dimensional space.
    max_edge_length:
        Maximum edge length to include in the Rips complex.
    max_dimension:
        Highest homological dimension to compute (1 = loops).

    Returns
    -------
    diag_h0 : ndarray, shape ``(n0, 2)``
        Persistence pairs ``[birth, death]`` for H0 (connected components).
        The single class with infinite death is clipped to *max_edge_length*.
    diag_h1 : ndarray, shape ``(n1, 2)``
        Persistence pairs for H1 (loops / 1-cycles).
    """
    if not _GUDHI_AVAILABLE:
        empty = np.zeros((0, 2), dtype=np.float32)
        return empty, empty

    rips = gudhi.RipsComplex(points=points, max_edge_length=max_edge_length)
    simplex_tree = rips.create_simplex_tree(max_dimension=max_dimension + 1)
    simplex_tree.compute_persistence()

    diag_h0_raw = simplex_tree.persistence_intervals_in_dimension(0)
    diag_h1_raw = simplex_tree.persistence_intervals_in_dimension(1)

    def _clean(raw: list, clip: float) -> np.ndarray:
        if len(raw) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        arr = np.array(raw, dtype=np.float32)
        # Replace +inf death values with clip value
        arr[~np.isfinite(arr)] = clip
        return arr

    return _clean(diag_h0_raw, max_edge_length), _clean(diag_h1_raw, max_edge_length)


def _detect_saddle_indices(
    points: np.ndarray,
    diag_h0: np.ndarray,
    max_edge_length: float = _MAX_FILTRATION,
) -> List[int]:
    """Identify "saddle" token indices.

    A saddle token is the token that *bridges* two previously disconnected
    components at the moment a 0-cycle dies (merges).  In Gudhi's elder rule
    the death of a 0-cycle corresponds to the shorter-lived component being
    merged; the edge responsible for the merge incident on that component's
    representative token is the saddle.

    We approximate this by finding, for each H0 persistence pair whose
    lifetime (death − birth) exceeds the median, the token closest to the
    midpoint of the pair's birth and death filtration radii.

    Parameters
    ----------
    points : ndarray, shape ``(L, D)``
    diag_h0 : ndarray, shape ``(n0, 2)``
    max_edge_length : float

    Returns
    -------
    List of token indices (integers in ``[0, L)``) identified as saddle
    points.  May be empty.
    """
    L = points.shape[0]
    if len(diag_h0) == 0 or L <= 1:
        # A single token has no nearest neighbour other than itself; skip.
        return []

    lifetimes = diag_h0[:, 1] - diag_h0[:, 0]
    if len(lifetimes) < 2:
        threshold = 0.0
    else:
        threshold = float(np.median(lifetimes))

    saddle_indices: List[int] = []
    dists = np.sum((points[:, None, :] - points[None, :, :]) ** 2, axis=-1)  # (L,L)

    for birth, death in diag_h0:
        lifetime = float(death - birth)
        if lifetime <= threshold:
            continue
        # The saddle radius is the death filtration value (merge radius)
        merge_radius_sq = float(death) ** 2
        # Find the token whose nearest-neighbour distance is closest to this
        # merge radius — it is the token "sitting at the saddle".
        # Use column index 1 to get the nearest *other* token (L >= 2 here).
        nn_dists = np.partition(dists, 1, axis=1)[:, 1]
        residual = np.abs(nn_dists - merge_radius_sq)
        saddle_idx = int(np.argmin(residual))
        if saddle_idx not in saddle_indices:
            saddle_indices.append(saddle_idx)

    return saddle_indices


# ---------------------------------------------------------------------------
# 2. Persistence Image Encoder (non-differentiable conversion to vector)
# ---------------------------------------------------------------------------

def _persistence_image(
    diagram: np.ndarray,
    resolution: int = _IMG_RESOLUTION,
    sigma: float = 0.1,
    max_val: float = _MAX_FILTRATION,
) -> np.ndarray:
    """Convert a persistence diagram to a flattened persistence image.

    The diagram is mapped to the (birth, persistence) plane where
    persistence = death − birth ≥ 0.  A Gaussian kernel is placed at each
    point and the result is evaluated on a ``resolution × resolution`` grid
    covering ``[0, max_val] × [0, max_val]``.  A linear weight that goes
    from 0 at persistence = 0 to 1 at persistence = max_val is applied to
    suppress near-diagonal noise.

    Parameters
    ----------
    diagram : ndarray, shape ``(n, 2)``
    resolution : int
    sigma : float  — bandwidth of the Gaussian kernel
    max_val : float — extent of the grid

    Returns
    -------
    ndarray of shape ``(resolution * resolution,)``
    """
    grid_axis = np.linspace(0.0, max_val, resolution)
    bx, by = np.meshgrid(grid_axis, grid_axis)   # (res, res) each
    img = np.zeros((resolution, resolution), dtype=np.float64)

    if len(diagram) == 0:
        return img.ravel().astype(np.float32)

    births = diagram[:, 0]
    deaths = diagram[:, 1]
    persistences = deaths - births  # >= 0 after clipping

    for b, p in zip(births, persistences):
        weight = p / max_val           # linear ramp — zero at diagonal
        gauss = np.exp(
            -((bx - b) ** 2 + (by - p) ** 2) / (2.0 * sigma ** 2)
        )
        img += weight * gauss

    return img.ravel().astype(np.float32)


# ---------------------------------------------------------------------------
# 3. Trainable modules
# ---------------------------------------------------------------------------

class TopologicalBiasNet(nn.Module):
    """Small MLP that maps a persistence image vector to an attention bias.

    Input:  concatenation of H0 and H1 persistence images  → size 2 * img_dim
    Output: scalar that is broadcast as an additive bias across all (L, L)
            query–key pairs, OR a (L, L) shaped matrix when *seq_len* is
            provided during the forward call.

    The MLP is the "Φ" in  Attention(Q,K,V) = Softmax(QK^T/√dk + Φ(PD)) V.
    """

    def __init__(self, img_dim: int, hidden_dim: int = 64) -> None:
        """
        Parameters
        ----------
        img_dim : int
            Size of a single persistence image vector (``resolution²`` for
            one homological dimension).  The input to the MLP will be
            ``2 * img_dim`` (H0 concatenated with H1).
        hidden_dim : int
            Number of hidden units in each intermediate layer of the MLP.
        """
        super().__init__()
        in_dim = 2 * img_dim           # H0 + H1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),  # single scalar bias (broadcast)
        )

    def forward(self, topo_vec: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        topo_vec : Tensor of shape ``(B, 2 * img_dim)``

        Returns
        -------
        Tensor of shape ``(B, 1, 1)`` — broadcast-ready attention bias.
        """
        return self.net(topo_vec).unsqueeze(-1)   # (B, 1, 1)


class SaddlePointEmbedding(nn.Module):
    """Learnable per-token bias injected at positions identified as saddles.

    For each saddle token detected by the frozen TDA layer, we add a
    learnable scalar offset to *all* attention logits in that token's row
    and column, encouraging the model to route information through saddle
    tokens.
    """

    def __init__(self) -> None:
        """Initialise the saddle-point embedding with a single learnable bias
        scalar.  The bias is initialised to zero so the module has no effect
        at the start of training and the model can learn whether and how much
        to emphasise saddle tokens.
        """
        super().__init__()
        self.saddle_bias = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        attn_logits: torch.Tensor,
        saddle_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        attn_logits : Tensor of shape ``(B, H, L, L)``
        saddle_mask : BoolTensor of shape ``(B, L)``
            True at positions that are saddle tokens.

        Returns
        -------
        Modified *attn_logits* of the same shape.
        """
        # Expand saddle_mask to (B, 1, L, 1) and (B, 1, 1, L) to add bias
        # to all rows *and* columns corresponding to saddle tokens.
        row_mask = saddle_mask.unsqueeze(1).unsqueeze(-1).float()   # (B,1,L,1)
        col_mask = saddle_mask.unsqueeze(1).unsqueeze(-2).float()   # (B,1,1,L)
        return attn_logits + self.saddle_bias * (row_mask + col_mask)


# ---------------------------------------------------------------------------
# 4. TopologicalAttention — the core module
# ---------------------------------------------------------------------------

class TopologicalAttention(nn.Module):
    """Multi-Head Attention augmented with Topological Data Analysis.

    Overrides standard scaled dot-product attention by adding a topological
    bias term derived from the Vietoris-Rips persistent homology of the input
    token point cloud.

    Formula
    -------
    .. code-block:: none

        Attention(Q, K, V) = Softmax( QK^T / sqrt(d_k) + Φ(PD) + S ) V

    where
      - Φ(PD) is the output of :class:`TopologicalBiasNet` (trainable MLP)
        applied to the persistence image of the current input batch.
      - S is the saddle-point bias (learnable scalar at detected saddle
        token positions, zero elsewhere).

    Parameters
    ----------
    d_model : int
        Total embedding dimension.
    num_heads : int
        Number of attention heads.  Must divide *d_model*.
    dropout : float
        Attention dropout probability.
    img_resolution : int
        Grid resolution for the persistence image (``img_resolution²``
        features per homological dimension).
    max_filtration : float
        Maximum Vietoris-Rips edge length.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.0,
        img_resolution: int = _IMG_RESOLUTION,
        max_filtration: float = _MAX_FILTRATION,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.img_resolution = img_resolution
        self.max_filtration = max_filtration

        img_dim = img_resolution * img_resolution   # e.g. 100

        # Standard QKV projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Topological bias network Φ
        self.topo_bias_net = TopologicalBiasNet(
            img_dim=img_dim, hidden_dim=max(32, d_model // 2)
        )

        # Saddle-point bias
        self.saddle_embed = SaddlePointEmbedding()

        self.dropout = nn.Dropout(p=dropout)
        self._scale = math.sqrt(self.d_k)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_topo_features(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run (non-differentiable) TDA on every sequence in the batch.

        Parameters
        ----------
        x : Tensor of shape ``(B, L, D)``

        Returns
        -------
        topo_vec : Tensor of shape ``(B, 2 * img_dim)``
            Concatenated H0 + H1 persistence images.
        saddle_mask : BoolTensor of shape ``(B, L)``
            True at positions identified as saddle tokens.
        """
        B, L, D = x.shape
        img_dim = self.img_resolution ** 2
        topo_vecs: List[np.ndarray] = []
        saddle_masks_np: List[np.ndarray] = []

        # Reduce dimensionality for TDA if D is large (cap at 32-D PCA)
        x_np = x.detach().cpu().float().numpy()   # (B, L, D)

        for b in range(B):
            pts = x_np[b]                         # (L, D)

            # Optional PCA projection when D > 32 to keep Rips tractable
            if D > 32:
                pts = _pca_project(pts, n_components=32)

            # Normalise point cloud to [0, 1]^D for stable filtration
            lo, hi = pts.min(), pts.max()
            span = hi - lo
            if span > 1e-8:
                pts_norm = (pts - lo) / span * (self.max_filtration * 0.5)
            else:
                pts_norm = pts

            diag_h0, diag_h1 = _compute_persistence_diagrams(
                pts_norm,
                max_edge_length=self.max_filtration,
                max_dimension=1,
            )

            img_h0 = _persistence_image(
                diag_h0,
                resolution=self.img_resolution,
                max_val=self.max_filtration,
            )
            img_h1 = _persistence_image(
                diag_h1,
                resolution=self.img_resolution,
                max_val=self.max_filtration,
            )
            topo_vecs.append(np.concatenate([img_h0, img_h1]))

            # Saddle mask
            saddle_idx = _detect_saddle_indices(pts_norm, diag_h0, self.max_filtration)
            mask = np.zeros(L, dtype=bool)
            for idx in saddle_idx:
                if 0 <= idx < L:
                    mask[idx] = True
            saddle_masks_np.append(mask)

        topo_tensor = torch.tensor(
            np.stack(topo_vecs, axis=0), dtype=x.dtype, device=x.device
        )  # (B, 2*img_dim)
        saddle_tensor = torch.tensor(
            np.stack(saddle_masks_np, axis=0), device=x.device
        )  # (B, L) bool
        return topo_tensor, saddle_tensor

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape ``(B, L, D)``
            Input token embeddings (latent representations).
        mask : BoolTensor of shape ``(B, L)`` or ``(B, L, L)``, optional
            Padding / causal mask.  True means "ignore this position."

        Returns
        -------
        Tensor of shape ``(B, L, D)``
        """
        B, L, D = x.shape

        # ---- 1. Standard QKV projections --------------------------------
        Q = self.W_q(x)   # (B, L, D)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape to (B, H, L, d_k)
        Q = Q.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.d_k).transpose(1, 2)

        # ---- 2. Scaled dot-product attention logits ---------------------
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / self._scale
        # shape: (B, H, L, L)

        # ---- 3. Topological bias Φ(PD) [frozen TDA + trainable MLP] ----
        topo_vec, saddle_mask = self._compute_topo_features(x)
        # topo_vec: (B, 2*img_dim), saddle_mask: (B, L)

        topo_bias = self.topo_bias_net(topo_vec)   # (B, 1, 1)
        topo_bias = topo_bias.unsqueeze(1)          # (B, 1, 1, 1) → broadcast
        attn_logits = attn_logits + topo_bias

        # ---- 4. Saddle-point bias S ------------------------------------
        attn_logits = self.saddle_embed(attn_logits, saddle_mask)

        # ---- 5. Optional padding/causal mask ----------------------------
        if mask is not None:
            if mask.dim() == 2:
                # (B, L) → (B, 1, 1, L)
                mask = mask.unsqueeze(1).unsqueeze(2)
            attn_logits = attn_logits.masked_fill(mask, float("-inf"))

        # ---- 6. Softmax + dropout + value aggregation -------------------
        attn_weights = F.softmax(attn_logits, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)       # (B, H, L, d_k)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(out)


# ---------------------------------------------------------------------------
# 5. TopologicalTransformerLayer
# ---------------------------------------------------------------------------

class TopologicalTransformerLayer(nn.Module):
    """Single transformer layer using :class:`TopologicalAttention`.

    Sub-layers
    ----------
    1. Topological multi-head self-attention  (pre-LN)
    2. Position-wise feed-forward network     (pre-LN)
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        img_resolution: int = _IMG_RESOLUTION,
        max_filtration: float = _MAX_FILTRATION,
    ) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.attn = TopologicalAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            img_resolution=img_resolution,
            max_filtration=max_filtration,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(p=dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Pre-LN residual blocks.

        Parameters
        ----------
        x : Tensor of shape ``(B, L, D)``
        mask : optional attention mask

        Returns
        -------
        Tensor of shape ``(B, L, D)``
        """
        # Self-attention block
        x = x + self.attn(self.norm1(x), mask=mask)
        # Feed-forward block
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# 6. TopologicalTransformer — full model
# ---------------------------------------------------------------------------

class TopologicalTransformer(nn.Module):
    """Stack of :class:`TopologicalTransformerLayer` blocks.

    This is the complete T-Form model.  It accepts raw token embeddings
    (or embeddings produced by an upstream encoder) and passes them through
    *num_layers* layers of topological self-attention followed by a final
    layer normalisation.

    Parameters
    ----------
    d_model : int
        Embedding / model dimension.
    num_heads : int
        Number of attention heads per layer.
    num_layers : int
        Number of stacked transformer layers.
    max_seq_len : int
        Maximum sequence length (used only for future positional encodings
        if needed; currently no positional encoding is applied — the TDA
        layer provides positional-structure information instead).
    d_ff : int, optional
        Feed-forward inner dimension.  Defaults to ``4 * d_model``.
    dropout : float
        Dropout probability applied in attention and FFN.
    img_resolution : int
        Persistence image grid resolution.
    max_filtration : float
        Maximum Vietoris-Rips filtration radius.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        max_seq_len: int = 512,
        d_ff: Optional[int] = None,
        dropout: float = 0.1,
        img_resolution: int = _IMG_RESOLUTION,
        max_filtration: float = _MAX_FILTRATION,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.layers = nn.ModuleList(
            [
                TopologicalTransformerLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    img_resolution=img_resolution,
                    max_filtration=max_filtration,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape ``(B, L, D)``
            Input latent embeddings.
        mask : optional attention mask (see :class:`TopologicalAttention`)

        Returns
        -------
        Tensor of shape ``(B, L, D)``
        """
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.norm(x)


# ---------------------------------------------------------------------------
# 7. Utility: lightweight PCA for high-dimensional point clouds
# ---------------------------------------------------------------------------

def _pca_project(points: np.ndarray, n_components: int = 32) -> np.ndarray:
    """Project *points* to its top *n_components* PCA directions.

    Uses the economy SVD so it is efficient even for large D.

    Parameters
    ----------
    points : ndarray of shape ``(L, D)``
    n_components : int

    Returns
    -------
    ndarray of shape ``(L, min(n_components, D))``
    """
    n_components = min(n_components, points.shape[0], points.shape[1])
    centred = points - points.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    return centred @ Vt[:n_components].T
