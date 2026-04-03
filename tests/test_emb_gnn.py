"""Tests for build_emb_gnn_profile (training-free GNN propagation)."""
import numpy as np
import pytest


def test_build_emb_gnn_profile_k1(graph_pt_path, tmp_path):
    """K=1 propagation, save only 2 models, check shapes."""
    from routeprofile import build_emb_gnn_profile

    save = str(tmp_path / "emb_gnn_k1.npz")
    build_emb_gnn_profile(
        graph=graph_pt_path,
        K=1,
        norm="sym",
        normalize=True,
        save=save,
        keep=["model-a", "model-b"],
    )
    result = np.load(save)
    assert set(result.files) == {"model-a", "model-b"}
    assert result["model-a"].shape == (768,)
    assert result["model-b"].shape == (768,)


def test_build_emb_gnn_profile_k2(graph_pt_path, tmp_path):
    """K=2 propagation with right-norm, save all models."""
    from routeprofile import build_emb_gnn_profile

    save = str(tmp_path / "emb_gnn_k2.npz")
    build_emb_gnn_profile(
        graph=graph_pt_path,
        K=2,
        norm="right",
        normalize=False,
        save=save,
        keep=[],    # empty list → save all models in graph
    )
    result = np.load(save)
    assert len(result.files) == 3   # all 3 model nodes saved


def test_build_emb_gnn_profile_norms(graph_pt_path, tmp_path):
    """All norm variants should run without error."""
    from routeprofile import build_emb_gnn_profile

    for norm in ("sym", "right", "left", "none"):
        save = str(tmp_path / f"emb_{norm}.npz")
        build_emb_gnn_profile(
            graph=graph_pt_path,
            K=1,
            norm=norm,
            save=save,
            keep=["model-a"],
        )
        result = np.load(save)
        assert "model-a" in result.files


def test_propagate_helper(tiny_task_graph):
    """Test the propagate() helper directly with the synthetic graph."""
    from routeprofile.get_model_profile.training_free.emb_gnn_profile import (
        propagate,
        save_model_embeddings,
        load_model_embeddings,
    )
    import torch, tempfile, os

    prop = propagate(tiny_task_graph, K=1, norm="sym", normalize=True)
    assert "model" in prop
    assert prop["model"].shape == (3, 768)

    # test save/load round-trip
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
        path = f.name
    try:
        save_model_embeddings(prop, tiny_task_graph, path, keep_names=["model-a", "model-b"])
        loaded = load_model_embeddings(path)
        assert set(loaded.keys()) == {"model-a", "model-b"}
        assert loaded["model-a"].shape == (768,)
    finally:
        os.unlink(path)
