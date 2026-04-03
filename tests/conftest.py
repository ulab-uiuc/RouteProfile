"""Shared fixtures for RouteProfile tests."""
import json
import os

import numpy as np
import pytest
import torch
from torch_geometric.data import HeteroData


@pytest.fixture
def tiny_task_graph():
    """
    Minimal HeteroData with arch(2) + model(3) + dataset(4) nodes.
    model→dataset edges carry benchmark scores (edge_attr).
    Both directions stored so HANConv and propagation can find neighbours.
    """
    data = HeteroData()

    data["arch"].x               = torch.randn(2, 768)
    data["arch"].node_names      = ["arch-0", "arch-1"]
    data["arch"].node_feature_text = ["Architecture A", "Architecture B"]

    data["model"].x              = torch.randn(3, 768)
    data["model"].node_names     = ["model-a", "model-b", "model-c"]
    data["model"].node_feature_text = ["Model A desc", "Model B desc", "Model C desc"]

    data["dataset"].x            = torch.randn(4, 768)
    data["dataset"].node_names   = ["ds-0", "ds-1", "ds-2", "ds-3"]
    data["dataset"].node_feature_text = ["DS0", "DS1", "DS2", "DS3"]

    # model → arch
    data["model", "belongs_to", "arch"].edge_index    = torch.tensor([[0, 1, 2], [0, 1, 0]])
    # arch → model (reverse)
    data["arch", "rev_belongs_to", "model"].edge_index = torch.tensor([[0, 1, 0], [0, 1, 2]])

    # model → dataset (with benchmark scores)
    ei_md = torch.tensor([[0, 0, 1, 1, 2], [0, 1, 1, 2, 2]])
    data["model", "evaluated_on", "dataset"].edge_index = ei_md
    data["model", "evaluated_on", "dataset"].edge_attr  = torch.rand(5, 1)
    # dataset → model (reverse)
    data["dataset", "rev_evaluated_on", "model"].edge_index = ei_md.flip(0)

    return data


@pytest.fixture
def mock_embeddings():
    """Three 768-dim unit vectors keyed by model name."""
    np.random.seed(0)
    embs = {
        "model-a": np.random.randn(768).astype("float32"),
        "model-b": np.random.randn(768).astype("float32"),
        "model-c": np.random.randn(768).astype("float32"),
    }
    # L2-normalise
    for k in embs:
        embs[k] /= np.linalg.norm(embs[k]) + 1e-8
    return embs


@pytest.fixture
def graph_pt_path(tiny_task_graph, tmp_path):
    """Save tiny_task_graph to a temp .pt file and return the path."""
    path = str(tmp_path / "graph.pt")
    torch.save(tiny_task_graph, path)
    return path


@pytest.fixture
def npz_profile_path(mock_embeddings, tmp_path):
    """Save mock_embeddings to a temp .npz file and return the path."""
    path = str(tmp_path / "profiles.npz")
    np.savez(path, **mock_embeddings)
    return path


@pytest.fixture
def routing_json_path(tmp_path):
    """
    Minimal routing_test_data.json with one query and three model entries.
    """
    data = [
        {
            "task_name": "test-task",
            "query": "What is 2+2?",
            "ground_truth": "4",
            "metric": "em",
            "choices": None,
            "model_performance": {
                "model-a": {"response": "4", "task_performance": 1.0, "success": True,
                            "prompt_tokens": 10, "completion_tokens": 1},
                "model-b": {"response": "3", "task_performance": 0.0, "success": True,
                            "prompt_tokens": 10, "completion_tokens": 1},
                "model-c": {"response": "4", "task_performance": 1.0, "success": True,
                            "prompt_tokens": 10, "completion_tokens": 1},
            },
        }
    ]
    path = str(tmp_path / "routing_test.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path
