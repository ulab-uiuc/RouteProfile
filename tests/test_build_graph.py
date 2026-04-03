"""Tests for all 5 build_*_graph functions using real profile_data/ JSON files.

get_longformer_embedding is mocked so no Longformer model is loaded.
The build functions default to profile_data/ paths; we only override save=.
"""
import importlib
import os
from unittest.mock import patch

import pytest
import torch

# ── Module handles (bypasses build_data_graph/__init__.py function shadowing) ─
_task_mod  = importlib.import_module("routeprofile.build_data_graph.build_task_graph")
_query_mod = importlib.import_module("routeprofile.build_data_graph.build_query_graph")
_qt_mod    = importlib.import_module("routeprofile.build_data_graph.build_query_task_graph")
_td_mod    = importlib.import_module("routeprofile.build_data_graph.build_task_domain_graph")
_qtd_mod   = importlib.import_module("routeprofile.build_data_graph.build_query_task_domain_graph")


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


# ── build_task_graph ─────────────────────────────────────────────────────────

def test_build_task_graph(tmp_path):
    """build_task_graph: produces a .pt with architecture/model/dataset nodes."""
    from routeprofile import build_task_graph

    save = str(tmp_path / "task_graph.pt")
    with patch.object(_task_mod, "get_longformer_embedding", side_effect=_mock_longformer):
        build_task_graph(mode="standard", save=save)

    assert os.path.exists(save)
    graph = torch.load(save, weights_only=False)
    assert "model"        in graph.node_types
    assert "dataset"      in graph.node_types
    assert "architecture" in graph.node_types


# ── build_query_graph ────────────────────────────────────────────────────────

def test_build_query_graph(tmp_path):
    """build_query_graph: adds query nodes on top of task graph."""
    from routeprofile import build_query_graph

    save = str(tmp_path / "query_graph.pt")
    with patch.object(_query_mod, "get_longformer_embedding", side_effect=_mock_longformer):
        build_query_graph(mode="standard", save=save)

    assert os.path.exists(save)
    graph = torch.load(save, weights_only=False)
    assert "model"        in graph.node_types
    assert "query"        in graph.node_types
    assert "architecture" in graph.node_types


# ── build_query_task_graph ───────────────────────────────────────────────────

def test_build_query_task_graph(tmp_path):
    """build_query_task_graph: query + architecture + model + dataset."""
    from routeprofile import build_query_task_graph

    save = str(tmp_path / "qt_graph.pt")
    with patch.object(_qt_mod, "get_longformer_embedding", side_effect=_mock_longformer):
        build_query_task_graph(mode="standard", save=save)

    assert os.path.exists(save)
    graph = torch.load(save, weights_only=False)
    assert "model"  in graph.node_types
    assert "query"  in graph.node_types


# ── build_task_domain_graph ──────────────────────────────────────────────────

def test_build_task_domain_graph(tmp_path):
    """build_task_domain_graph: adds domain nodes to task graph."""
    from routeprofile import build_task_domain_graph

    save = str(tmp_path / "td_graph.pt")
    with patch.object(_td_mod, "get_longformer_embedding", side_effect=_mock_longformer):
        build_task_domain_graph(mode="standard", save=save)

    assert os.path.exists(save)
    graph = torch.load(save, weights_only=False)
    assert "model"  in graph.node_types
    assert "domain" in graph.node_types
    assert "dataset" in graph.node_types


# ── build_query_task_domain_graph ────────────────────────────────────────────

def test_build_query_task_domain_graph(tmp_path):
    """build_query_task_domain_graph: full graph with all 5 node types."""
    from routeprofile import build_query_task_domain_graph

    save = str(tmp_path / "qtd_graph.pt")
    with patch.object(_qtd_mod, "get_longformer_embedding", side_effect=_mock_longformer):
        build_query_task_domain_graph(mode="standard", save=save)

    assert os.path.exists(save)
    graph = torch.load(save, weights_only=False)
    assert "model"        in graph.node_types
    assert "query"        in graph.node_types
    assert "domain"       in graph.node_types
    assert "architecture" in graph.node_types
    assert "dataset"      in graph.node_types
