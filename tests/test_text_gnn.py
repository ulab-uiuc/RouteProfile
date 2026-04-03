"""Tests for build_text_gnn_profile (text_gnn_profile.py).

K=0 skips vLLM entirely and encodes raw node_feature_text via Longformer.
We mock get_longformer_embedding so no real model is loaded.
GPU 9 / gpu_memory_utilization are relevant for K>0 production calls.
"""
import importlib
import json
import os
from unittest.mock import patch

import numpy as np
import pytest
import torch

# Get the actual module (bypasses training_free/__init__.py alias shadowing)
_text_mod = importlib.import_module(
    "routeprofile.get_model_profile.training_free.text_gnn_profile"
)


def _mock_longformer(texts, **kwargs):
    """Return L2-normalised random tensors matching real get_longformer_embedding."""
    if isinstance(texts, str):
        v = torch.randn(768)
        return v / v.norm()
    n = len(texts) if hasattr(texts, "__len__") else 1
    if n == 1:
        v = torch.randn(768)
        return v / v.norm()
    v = torch.randn(n, 768)
    return v / v.norm(dim=1, keepdim=True)


# ── Import smoke test ─────────────────────────────────────────────────────────

def test_build_text_gnn_importable():
    """build_text_gnn_profile is importable and callable."""
    from routeprofile import build_text_gnn_profile
    assert callable(build_text_gnn_profile)


# ── K=0: Longformer-only baseline (no vLLM) ───────────────────────────────────

def test_build_text_gnn_k0(graph_pt_path, tmp_path):
    """K=0 encodes raw node_feature_text via Longformer (mocked); saves npz + json."""
    from routeprofile import build_text_gnn_profile

    emb_save  = str(tmp_path / "text_gnn.npz")
    text_save = str(tmp_path / "text_gnn_texts.json")

    with patch.object(_text_mod, "get_longformer_embedding", side_effect=_mock_longformer):
        build_text_gnn_profile(
            graph=graph_pt_path,
            K=0,
            emb_save=emb_save,
            text_save=text_save,
            keep=[],          # keep=[] → save ALL model nodes in the graph
        )

    # embeddings file
    assert os.path.exists(emb_save), "Embedding .npz not created"
    result = np.load(emb_save)
    assert len(result.files) > 0, "No model embeddings saved"
    assert result[result.files[0]].shape == (768,), "Unexpected embedding shape"

    # texts file
    assert os.path.exists(text_save), "Text .json not created"
    with open(text_save) as f:
        texts = json.load(f)
    assert "final_texts" in texts
    assert "hop_texts" in texts
    assert len(texts["final_texts"]) > 0


def test_build_text_gnn_k0_model_names(graph_pt_path, tmp_path):
    """K=0 with keep=[]: saved model names match those in the tiny_task_graph fixture."""
    from routeprofile import build_text_gnn_profile

    emb_save = str(tmp_path / "text_gnn.npz")

    with patch.object(_text_mod, "get_longformer_embedding", side_effect=_mock_longformer):
        build_text_gnn_profile(
            graph=graph_pt_path,
            K=0,
            emb_save=emb_save,
            text_save=str(tmp_path / "texts.json"),
            keep=[],
        )

    result = np.load(emb_save)
    saved_names = set(result.files)
    # tiny_task_graph has 3 model nodes named model-a, model-b, model-c
    assert saved_names == {"model-a", "model-b", "model-c"}
