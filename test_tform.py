"""Tests for the Topological Transformer (T-Form) implementation."""

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from tform import (
    TopologicalAttention,
    TopologicalTransformer,
    TopologicalTransformerLayer,
    _compute_persistence_diagrams,
    _detect_saddle_indices,
    _pca_project,
    _persistence_image,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_batch(B: int = 2, L: int = 8, D: int = 16) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(B, L, D)


# ---------------------------------------------------------------------------
# Unit tests — TDA utilities
# ---------------------------------------------------------------------------


class TestPersistenceDiagrams:
    def test_output_shapes(self):
        pts = np.random.randn(10, 4).astype(np.float32)
        h0, h1 = _compute_persistence_diagrams(pts)
        assert h0.ndim == 2 and h0.shape[1] == 2
        assert h1.ndim == 2 and h1.shape[1] == 2

    def test_no_infinite_values(self):
        pts = np.random.randn(8, 3).astype(np.float32)
        h0, h1 = _compute_persistence_diagrams(pts, max_edge_length=5.0)
        assert np.all(np.isfinite(h0))
        assert np.all(np.isfinite(h1))

    def test_birth_leq_death(self):
        pts = np.random.randn(12, 2).astype(np.float32)
        h0, h1 = _compute_persistence_diagrams(pts)
        if len(h0):
            assert np.all(h0[:, 0] <= h0[:, 1])
        if len(h1):
            assert np.all(h1[:, 0] <= h1[:, 1])

    def test_empty_diagram_on_single_point(self):
        pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        h0, h1 = _compute_persistence_diagrams(pts)
        # Single point — no 1-cycles possible
        assert h1.shape[0] == 0


class TestPersistenceImage:
    def test_output_length(self):
        diagram = np.array([[0.0, 1.0], [0.5, 2.0]], dtype=np.float32)
        img = _persistence_image(diagram, resolution=5)
        assert img.shape == (25,)

    def test_empty_diagram_gives_zeros(self):
        img = _persistence_image(np.zeros((0, 2), dtype=np.float32), resolution=5)
        assert np.all(img == 0.0)

    def test_nonnegative(self):
        np.random.seed(0)
        diagram = np.abs(np.random.randn(20, 2).astype(np.float32))
        diagram[:, 1] += diagram[:, 0]
        img = _persistence_image(diagram)
        assert np.all(img >= 0.0)


class TestSaddleDetection:
    def test_returns_list(self):
        pts = np.random.randn(10, 4).astype(np.float32)
        h0, _ = _compute_persistence_diagrams(pts)
        saddles = _detect_saddle_indices(pts, h0)
        assert isinstance(saddles, list)

    def test_indices_in_range(self):
        pts = np.random.randn(15, 3).astype(np.float32)
        h0, _ = _compute_persistence_diagrams(pts)
        saddles = _detect_saddle_indices(pts, h0)
        L = pts.shape[0]
        assert all(0 <= i < L for i in saddles)

    def test_empty_diagram_gives_empty(self):
        pts = np.random.randn(5, 2).astype(np.float32)
        saddles = _detect_saddle_indices(pts, np.zeros((0, 2), dtype=np.float32))
        assert saddles == []


class TestPCAProject:
    def test_output_shape(self):
        pts = np.random.randn(20, 128).astype(np.float32)
        proj = _pca_project(pts, n_components=16)
        assert proj.shape == (20, 16)

    def test_does_not_exceed_input_rank(self):
        pts = np.random.randn(5, 100).astype(np.float32)
        proj = _pca_project(pts, n_components=50)
        # Can't have more components than min(L, D) = 5
        assert proj.shape[1] <= min(5, 100)


# ---------------------------------------------------------------------------
# Integration tests — TopologicalAttention
# ---------------------------------------------------------------------------


class TestTopologicalAttention:
    def test_output_shape(self):
        B, L, D = 2, 8, 32
        attn = TopologicalAttention(d_model=D, num_heads=4)
        x = make_batch(B, L, D)
        out = attn(x)
        assert out.shape == (B, L, D)

    def test_output_is_finite(self):
        attn = TopologicalAttention(d_model=16, num_heads=2)
        x = make_batch(2, 6, 16)
        out = attn(x)
        assert torch.all(torch.isfinite(out))

    def test_gradients_flow(self):
        """The trainable parts (QKV projections, topo bias MLP) must have
        non-zero gradients even though TDA is frozen."""
        attn = TopologicalAttention(d_model=16, num_heads=2)
        x = make_batch(1, 6, 16)
        out = attn(x)
        loss = out.sum()
        loss.backward()
        # Check at least one trainable parameter received a gradient
        trained_params = [p for p in attn.parameters() if p.grad is not None]
        assert len(trained_params) > 0

    def test_with_padding_mask(self):
        B, L, D = 2, 8, 32
        attn = TopologicalAttention(d_model=D, num_heads=4)
        x = make_batch(B, L, D)
        # Mask the last 2 positions for all batch items
        mask = torch.zeros(B, L, dtype=torch.bool)
        mask[:, -2:] = True
        out = attn(x, mask=mask)
        assert out.shape == (B, L, D)

    def test_invalid_d_model_raises(self):
        with pytest.raises(ValueError):
            TopologicalAttention(d_model=17, num_heads=4)

    def test_single_token(self):
        """Edge case: single token sequence."""
        attn = TopologicalAttention(d_model=16, num_heads=2)
        x = make_batch(1, 1, 16)
        out = attn(x)
        assert out.shape == (1, 1, 16)


# ---------------------------------------------------------------------------
# Integration tests — TopologicalTransformerLayer
# ---------------------------------------------------------------------------


class TestTopologicalTransformerLayer:
    def test_output_shape(self):
        B, L, D = 2, 10, 32
        layer = TopologicalTransformerLayer(d_model=D, num_heads=4, dropout=0.0)
        x = make_batch(B, L, D)
        out = layer(x)
        assert out.shape == (B, L, D)

    def test_residual_connection_changes_output(self):
        """The output should differ from the input (i.e. something happened)."""
        layer = TopologicalTransformerLayer(d_model=16, num_heads=2, dropout=0.0)
        x = make_batch(1, 6, 16)
        out = layer(x)
        assert not torch.allclose(out, x)


# ---------------------------------------------------------------------------
# Integration tests — TopologicalTransformer (full model)
# ---------------------------------------------------------------------------


class TestTopologicalTransformer:
    def test_output_shape(self):
        B, L, D = 2, 12, 32
        model = TopologicalTransformer(
            d_model=D, num_heads=4, num_layers=2, max_seq_len=64, dropout=0.0
        )
        x = make_batch(B, L, D)
        out = model(x)
        assert out.shape == (B, L, D)

    def test_deterministic_in_eval_mode(self):
        model = TopologicalTransformer(
            d_model=16, num_heads=2, num_layers=1, dropout=0.0
        )
        model.eval()
        x = make_batch(1, 8, 16)
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_batch_size_one(self):
        model = TopologicalTransformer(
            d_model=16, num_heads=2, num_layers=1, dropout=0.0
        )
        x = make_batch(1, 8, 16)
        out = model(x)
        assert out.shape == (1, 8, 16)

    def test_varying_sequence_length(self):
        model = TopologicalTransformer(
            d_model=16, num_heads=2, num_layers=1, dropout=0.0
        )
        for L in (4, 8, 16):
            x = make_batch(1, L, 16)
            out = model(x)
            assert out.shape == (1, L, 16), f"Failed for L={L}"

    def test_parameter_count_reasonable(self):
        """Just a sanity check that the model has learnable parameters."""
        model = TopologicalTransformer(
            d_model=32, num_heads=4, num_layers=2, dropout=0.0
        )
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

    def test_output_finite(self):
        model = TopologicalTransformer(
            d_model=16, num_heads=2, num_layers=2, dropout=0.0
        )
        x = make_batch(2, 8, 16)
        out = model(x)
        assert torch.all(torch.isfinite(out))

    def test_high_dimensional_input(self):
        """When D > 32, PCA projection is applied before TDA."""
        model = TopologicalTransformer(
            d_model=64, num_heads=4, num_layers=1, dropout=0.0
        )
        x = make_batch(1, 8, 64)
        out = model(x)
        assert out.shape == (1, 8, 64)
