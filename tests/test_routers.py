"""Tests for call_simrouter, call_mlprouter, call_graphrouter."""
import importlib
import json
import os
from unittest.mock import patch

import numpy as np
import pytest
import torch


# Grab the actual router modules (bypasses __init__.py name shadowing)
_sim_mod   = importlib.import_module("routeprofile.routing_evaluation.SimRouter")
_mlp_mod   = importlib.import_module("routeprofile.routing_evaluation.MLPRouter")
_graph_mod = importlib.import_module("routeprofile.routing_evaluation.GraphRouter")


# ── Mock helper ────────────────────────────────────────────────────────────────

def _mock_longformer(texts, batch_size=32):
    """Return random 768-dim tensors matching the real get_longformer_embedding signature."""
    if isinstance(texts, str):
        return torch.randn(768)
    n = len(texts) if hasattr(texts, "__len__") else 1
    if n == 1:
        return torch.randn(768)
    return torch.randn(n, 768)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def train_json_path(tmp_path):
    """Minimal pairwise training data (routing_train_data format)."""
    data = [
        {
            "task_name": "test-task",
            "query": "What is 2+2?",
            "better_model": "model-a",
            "worse_model":  "model-b",
            "choices": None,
        },
        {
            "task_name": "test-task",
            "query": "What is 3+3?",
            "better_model": "model-c",
            "worse_model":  "model-b",
            "choices": None,
        },
    ]
    path = str(tmp_path / "routing_train.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


# ── SimRouter ─────────────────────────────────────────────────────────────────

def test_call_simrouter(npz_profile_path, routing_json_path, tmp_path):
    """call_simrouter should produce a results JSON with routing_results."""
    from routeprofile import call_simrouter

    out = str(tmp_path / "sim_results.json")

    with patch.object(_sim_mod, "get_longformer_embedding", side_effect=_mock_longformer):
        result = call_simrouter(
            model_profile_path=npz_profile_path,
            routing_data_path=routing_json_path,
            output_path=out,
        )

    assert os.path.exists(out), "Output file not created"
    with open(out) as f:
        saved = json.load(f)
    assert "routing_results" in saved
    assert len(saved["routing_results"]) == 1
    assert saved["routing_results"][0]["model_name"] in {"model-a", "model-b", "model-c"}


def test_simrouter_class_importable():
    """SimRouter class can be imported from the routing_evaluation sub-package."""
    from routeprofile.routing_evaluation import SimRouter
    assert SimRouter is not None


# ── MLPRouter ─────────────────────────────────────────────────────────────────

def test_call_mlprouter(npz_profile_path, train_json_path, routing_json_path, tmp_path):
    """call_mlprouter should train for 2 epochs and produce a results JSON."""
    from routeprofile import call_mlprouter

    out      = str(tmp_path / "mlp_results.json")
    save_ckpt = str(tmp_path / "mlp_ckpt.pt")

    with patch.object(_mlp_mod, "get_longformer_embedding", side_effect=_mock_longformer):
        call_mlprouter(
            model_profile_path=npz_profile_path,
            training_data_path=train_json_path,
            testing_data_path=routing_json_path,
            output_path=out,
            save_ckpt=save_ckpt,
            epochs=2,
            batch_size=2,
        )

    assert os.path.exists(out), "Output file not created"
    with open(out) as f:
        saved = json.load(f)
    assert "routing_results" in saved


def test_mlprouter_class_importable():
    """MLPRouter PyTorch module is importable."""
    from routeprofile.routing_evaluation import MLPRouter
    # Can instantiate a minimal version
    router = MLPRouter(in_dim=768, hidden_dim=64, out_dim=32, num_layers=2, dropout=0.0)
    x = torch.randn(2, 768)
    q_emb = router.encode_query(x)
    assert q_emb.shape == (2, 32)


# ── GraphRouter ───────────────────────────────────────────────────────────────

def test_graphrouter_class_importable():
    """GraphRouter class renamed correctly from GraphRouterModel."""
    from routeprofile.routing_evaluation import GraphRouter
    assert GraphRouter is not None
    # Verify old name is gone
    import routeprofile.routing_evaluation.GraphRouter as gr_mod
    assert not hasattr(gr_mod, "GraphRouterModel"), \
        "GraphRouterModel should have been renamed to GraphRouter"


def test_call_graphrouter(npz_profile_path, train_json_path, routing_json_path, tmp_path):
    """call_graphrouter should train for 2 epochs and produce a results JSON."""
    from routeprofile import call_graphrouter

    out       = str(tmp_path / "graph_results.json")
    save_ckpt = str(tmp_path / "graph_ckpt.pt")

    with patch.object(_graph_mod, "get_longformer_embedding", side_effect=_mock_longformer):
        call_graphrouter(
            model_profile_path=npz_profile_path,
            training_data_path=train_json_path,
            testing_data_path=routing_json_path,
            output_path=out,
            save_ckpt=save_ckpt,
            epochs=2,
            batch_size=2,
        )

    assert os.path.exists(out), "Output file not created"
    with open(out) as f:
        saved = json.load(f)
    assert "routing_results" in saved
