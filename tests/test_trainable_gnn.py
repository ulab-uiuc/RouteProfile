"""Tests for build_trainable_gnn_profile (HANConv self-supervised pretraining)."""
import numpy as np
import pytest
import torch


def test_hanencoder_forward(tiny_task_graph):
    """HANEncoder forward pass returns model embeddings of the right shape."""
    from routeprofile.get_model_profile.trainable.trainable_gnn_profile import HANEncoder

    encoder = HANEncoder(
        in_channels=768,
        hidden_dim=64,
        out_dim=768,
        heads=2,
        num_layers=2,
        metadata=tiny_task_graph.metadata(),
    )
    x_dict = {ntype: tiny_task_graph[ntype].x for ntype in tiny_task_graph.node_types}
    ei_dict = {
        etype: tiny_task_graph[etype].edge_index
        for etype in tiny_task_graph.edge_types
    }
    out = encoder(x_dict, ei_dict)

    assert "model" in out
    assert out["model"].shape == (3, 768)
    # L2-normalised
    norms = torch.linalg.norm(out["model"], dim=1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-5)


def test_hanencoder_single_layer(tiny_task_graph):
    """Single-layer HANEncoder should also work."""
    from routeprofile.get_model_profile.trainable.trainable_gnn_profile import HANEncoder

    encoder = HANEncoder(
        in_channels=768, hidden_dim=64, out_dim=768, heads=1,
        num_layers=1, metadata=tiny_task_graph.metadata(),
    )
    x_dict = {n: tiny_task_graph[n].x for n in tiny_task_graph.node_types}
    ei_dict = {e: tiny_task_graph[e].edge_index for e in tiny_task_graph.edge_types}
    out = encoder(x_dict, ei_dict)
    assert out["model"].shape == (3, 768)


def test_build_trainable_gnn_profile_runs(graph_pt_path, tmp_path):
    """Run 2 training epochs on the synthetic graph; check outputs exist."""
    from routeprofile import build_trainable_gnn_profile

    emb_path  = str(tmp_path / "trainable.npz")
    ckpt_path = str(tmp_path / "ckpt.pt")

    build_trainable_gnn_profile(
        graph=graph_pt_path,
        save_emb=emb_path,
        save_ckpt=ckpt_path,
        hidden_dim=64,
        out_dim=768,
        heads=2,
        num_layers=2,
        node_mask_rate=0.3,
        edge_mask_rate=0.3,
        edge_loss_w=1.0,
        lr=1e-3,
        epochs=2,
        seed=42,
        keep=["model-a"],
    )

    # embedding file
    result = np.load(emb_path)
    assert "model-a" in result.files
    assert result["model-a"].shape == (768,)

    # checkpoint file
    ckpt = torch.load(ckpt_path, weights_only=False)
    assert "encoder" in ckpt
    assert "config" in ckpt
    assert ckpt["config"]["out_dim"] == 768


def test_build_trainable_gnn_keep_all(graph_pt_path, tmp_path):
    """keep=[] means save all models in the graph."""
    from routeprofile import build_trainable_gnn_profile

    emb_path  = str(tmp_path / "all.npz")
    ckpt_path = str(tmp_path / "ckpt_all.pt")
    build_trainable_gnn_profile(
        graph=graph_pt_path,
        save_emb=emb_path,
        save_ckpt=ckpt_path,
        hidden_dim=32, out_dim=768, heads=1, num_layers=1,
        epochs=1, seed=0, keep=[],
    )
    result = np.load(emb_path)
    # All 3 model nodes should be present
    assert len(result.files) == 3
