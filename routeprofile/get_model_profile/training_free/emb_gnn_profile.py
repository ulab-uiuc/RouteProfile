"""
Training-free GNN for heterogeneous graphs via simple neighbourhood propagation.

The aggregation rule per hop is:

    X_dst  ←  A_norm · X_src

where A_norm is the degree-normalised adjacency for each edge type.
Repeating this K times gives:

    K=1 :  A · X
    K=2 :  A · (A · X)
    ...

For a heterogeneous graph each node type accumulates messages from all its
incoming edge types and the results are mean-pooled across relation types,
which keeps every node type at the same feature dimension (768) throughout.

Normalisation choices (controlled by `norm`):
  "sym"   – symmetric,  D_dst^{-1/2} · A · D_src^{-1/2}   (default)
  "right" – right-only, A · D_src^{-1}
  "left"  – left-only,  D_dst^{-1} · A
  "none"  – raw adjacency, no normalisation

Edge feature weighting (model <-> dataset edges only):
  When an edge carries an `edge_attr` (benchmark score), the message is
  additionally scaled by a normalised version of that score:

      msg[e] = norm_weight[e] · norm_score[e] · x_src[e]

  Scores are min-max normalised per edge type to [0, 1] so that the scale
  of benchmark numbers does not dominate the structural normalisation.
  Edges without `edge_attr` (arch <-> model) are unaffected.

Usage example
-------------
    data   = torch.load("llm_hetero_graph.pt", weights_only=False)
    output = propagate(data, K=2)

    # output is a dict { node_type: FloatTensor [N, 768] }
    model_embeddings = output["model"]        # shape [9, 768]
    arch_embeddings  = output["architecture"] # shape [4, 768]
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.utils import degree


# ── Core propagation ───────────────────────────────────────────────────────────

def _build_norm_weights(
    edge_index: Tensor,
    num_src:    int,
    num_dst:    int,
    norm:       str,
) -> Tensor:
    """
    Compute a per-edge scalar normalisation weight.

    Args:
        edge_index : LongTensor [2, E]  (src, dst) indices
        num_src    : number of source nodes
        num_dst    : number of destination nodes
        norm       : one of {"sym", "right", "left", "none"}

    Returns:
        FloatTensor [E] of edge weights.
    """
    src_idx, dst_idx = edge_index[0], edge_index[1]

    if norm == "none":
        return torch.ones(edge_index.size(1))

    # out-degree of source nodes, in-degree of destination nodes
    deg_src = degree(src_idx, num_nodes=num_src)   # [num_src]
    deg_dst = degree(dst_idx, num_nodes=num_dst)   # [num_dst]

    # replace zeros to avoid division by zero
    deg_src = deg_src.clamp(min=1.0)
    deg_dst = deg_dst.clamp(min=1.0)

    if norm == "right":
        # A · D_src^{-1}
        weights = 1.0 / deg_src[src_idx]
    elif norm == "left":
        # D_dst^{-1} · A
        weights = 1.0 / deg_dst[dst_idx]
    elif norm == "sym":
        # D_dst^{-1/2} · A · D_src^{-1/2}
        weights = deg_dst[dst_idx].pow(-0.5) * deg_src[src_idx].pow(-0.5)
    else:
        raise ValueError(f"Unknown norm '{norm}'. Choose from: sym, right, left, none.")

    return weights   # [E]


def _normalise_edge_attr(edge_attr: Tensor) -> Tensor:
    """
    Min-max normalise edge attributes to [0, 1] across all edges of one type.

    This prevents raw benchmark score magnitudes (e.g. 0–100) from
    overwhelming the structural normalisation weights.

    Args:
        edge_attr : FloatTensor [E, 1]

    Returns:
        FloatTensor [E, 1] with values in [0, 1].
        If all values are identical the tensor is returned as all-ones.
    """
    scores = edge_attr.squeeze(1)          # [E]
    s_min, s_max = scores.min(), scores.max()
    if (s_max - s_min).abs() < 1e-8:
        return torch.ones_like(scores).unsqueeze(1)
    return ((scores - s_min) / (s_max - s_min)).unsqueeze(1)   # [E, 1]


def _propagate_one_hop(
    node_feats:  dict[str, Tensor],
    data:        HeteroData,
    norm:        str,
) -> dict[str, Tensor]:
    """
    Execute one round of message passing across all edge types.

    For every edge type (src_type, rel, dst_type):

        # edges WITHOUT edge_attr  (e.g. arch <-> model)
        msg[e] = norm_weight[e] · x_src[e]

        # edges WITH edge_attr  (e.g. model <-> dataset, score-weighted)
        msg[e] = norm_weight[e] · norm_score[e] · x_src[e]

        out_dst += scatter_add(msg, dst_idx)

    Destination nodes that receive messages from multiple relation types
    have their contributions mean-pooled.

    Args:
        node_feats : current feature dict { node_type: Tensor [N, D] }
        data       : the original HeteroData (for edge_index, edge_attr, sizes)
        norm       : normalisation mode

    Returns:
        Updated feature dict with the same keys and shapes.
    """
    feat_dim = next(iter(node_feats.values())).size(1)

    # accumulators: sum of aggregated messages and count of contributing relations
    agg_sum   = {ntype: torch.zeros(data[ntype].num_nodes, feat_dim)
                 for ntype in node_feats}
    agg_count = {ntype: 0 for ntype in node_feats}

    for (src_type, rel, dst_type), edge_store in data.edge_items():
        if "edge_index" not in edge_store:
            continue

        edge_index = edge_store["edge_index"]          # [2, E]
        x_src      = node_feats[src_type]              # [N_src, D]
        num_src    = data[src_type].num_nodes
        num_dst    = data[dst_type].num_nodes
        E          = edge_index.size(1)

        # per-edge structural normalisation weights  [E]
        weights = _build_norm_weights(edge_index, num_src, num_dst, norm)

        # per-edge score weights  [E, 1]  (ones if no edge_attr)
        has_attr    = hasattr(edge_store, "edge_attr") and edge_store.edge_attr is not None
        score_scale = _normalise_edge_attr(edge_store.edge_attr) if has_attr                       else torch.ones(E, 1)

        src_idx, dst_idx = edge_index[0], edge_index[1]

        # combine structural weight and score weight, then scale source features
        # weights: [E] -> [E, 1];  score_scale: [E, 1]  =>  combined: [E, 1]
        combined = weights.unsqueeze(1) * score_scale        # [E, 1]
        msgs     = x_src[src_idx] * combined                 # [E, D]

        # scatter-add into destination buffer
        agg_sum[dst_type].scatter_add_(
            0,
            dst_idx.unsqueeze(1).expand(E, feat_dim),
            msgs,
        )
        agg_count[dst_type] += 1

    # mean-pool across relation types; keep nodes with no incoming edges unchanged
    new_feats: dict[str, Tensor] = {}
    for ntype in node_feats:
        if agg_count[ntype] > 0:
            new_feats[ntype] = agg_sum[ntype] / agg_count[ntype]
        else:
            # isolated node type: carry forward unchanged
            new_feats[ntype] = node_feats[ntype].clone()

    return new_feats


def propagate(
    data:      HeteroData,
    K:         int  = 1,
    norm:      str  = "sym",
    normalize: bool = False,
) -> dict[str, Tensor]:
    """
    Training-free neighbourhood propagation: repeatedly apply A · X for K hops.

    Args:
        data      : HeteroData graph with `.x` tensors on every node type.
        K         : number of propagation hops (K=0 returns raw features).
        norm      : edge weight normalisation — one of:
                      "sym"   symmetric  D^{-1/2} A D^{-1/2}  (default)
                      "right" row-wise   A D^{-1}
                      "left"  col-wise   D^{-1} A
                      "none"  no normalisation
        normalize : if True, L2-normalise output embeddings per node.

    Returns:
        dict { node_type: FloatTensor [N, D] } with propagated features.
    """
    if K < 0:
        raise ValueError(f"K must be >= 0, got {K}.")

    # initialise from graph node features
    node_feats: dict[str, Tensor] = {
        ntype: data[ntype].x.float()
        for ntype in data.node_types
    }

    for hop in range(K):
        node_feats = _propagate_one_hop(node_feats, data, norm)
        print(f"  Hop {hop + 1}/{K} done.")

    if normalize:
        node_feats = {
            ntype: F.normalize(x, p=2, dim=1)
            for ntype, x in node_feats.items()
        }

    return node_feats


# ── Summary helper ─────────────────────────────────────────────────────────────

def print_propagation_summary(
    before: dict[str, Tensor],
    after:  dict[str, Tensor],
    K:      int,
    norm:   str,
) -> None:
    """Print a compact before/after comparison of node embedding norms."""
    print(f"\n=== Propagation summary  (K={K}, norm='{norm}') ===")
    print(f"  {'node type':<15} {'shape':<15} {'mean |x| before':>16}  {'mean |x| after':>15}")
    print("  " + "-" * 65)
    for ntype in before:
        b_norm = before[ntype].norm(dim=1).mean().item()
        a_norm = after[ntype].norm(dim=1).mean().item()
        shape  = list(after[ntype].shape)
        print(f"  {ntype:<15} {str(shape):<15} {b_norm:>16.4f}  {a_norm:>15.4f}")


# ── Save / load helpers ────────────────────────────────────────────────────────

def save_model_embeddings(
    prop_feats:  dict[str, Tensor],
    data:        HeteroData,
    save_path:   str = "model_embeddings.npz",
    keep_names:  list[str] | None = None,
) -> None:
    """
    Extract model node embeddings and save as a .npz archive.

    The archive maps each model's node_name (string key) to its embedding
    vector stored as a float32 numpy array of shape [D].

    Args:
        prop_feats  : output of propagate(), { node_type: Tensor [N, D] }
        data        : the original HeteroData (needed for model node_names)
        save_path   : output file path (should end in .npz)
        keep_names  : optional list of model name keys to save.
                      If None, all models are saved.
                      Names not found in the graph are warned and skipped.

    Example — loading back:
        loaded   = np.load("model_embeddings.npz")
        emb_dict = { name: loaded[name] for name in loaded.files }
    """
    model_tensor: Tensor    = prop_feats["model"]          # [N_models, D]
    model_names:  list[str] = data["model"].node_names     # length N_models

    # build a name → index lookup for fast filtering
    name2idx: dict[str, int] = {name: i for i, name in enumerate(model_names)}

    if keep_names is not None:
        # warn about any requested names that are absent from the graph
        missing = [n for n in keep_names if n not in name2idx]
        if missing:
            print(f"  Warning: the following requested model names were not found "
                  f"in the graph and will be skipped: {missing}")
        selected = [(name, name2idx[name]) for name in keep_names if name in name2idx]
    else:
        selected = list(enumerate(model_names))          # (idx, name)
        selected = [(name, idx) for idx, name in enumerate(model_names)]

    emb_dict: dict[str, np.ndarray] = {
        name: model_tensor[idx].numpy().astype(np.float32)
        for name, idx in selected
    }

    np.savez(save_path, **emb_dict)

    saved_names = list(emb_dict.keys())
    print(f"  Saved {len(emb_dict)} model embeddings → '{save_path}'")
    print(f"  Embedding dim : {model_tensor.size(1)}")
    print(f"  Models stored : {saved_names}")
    print( "  Load with     : loaded = np.load('model_embeddings.npz')")
    print( "                  emb_dict = { k: loaded[k] for k in loaded.files }")


def load_model_embeddings(npz_path: str) -> dict[str, np.ndarray]:
    """
    Load model embeddings saved by save_model_embeddings().

    Args:
        npz_path : path to the .npz file

    Returns:
        dict { model_name: np.ndarray [D] }
    """
    loaded = np.load(npz_path)
    return { name: loaded[name] for name in loaded.files }


# ── Main (demo) ────────────────────────────────────────────────────────────────

# Models to extract from the large graph.
# Set to None to save all models.
TARGET_MODELS: list[str] | None = [
    "qwen2.5-7b-instruct",
    "gemma-2-9b-it",
    "llama-3.1-8b-instruct",
    "mixtral-8x7b-instruct-v0.1",
    "mixtral-8x22b-instruct-v0.1",
    "llama-3.2-3b-instruct",
    "mistral-small-24b-instruct-2501-bf16",
    "llama-3.3-70b-instruct",
]


def build_model_profile(
    mode:      str            = "standard",
    graph:     str | None     = None,
    K:         int            = 2,
    norm:      str            = "sym",
    normalize: bool           = False,
    save:      str | None     = None,
    keep:      list | None    = None,
) -> None:
    """
    Load a saved HeteroData graph, run K-hop propagation, and save model
    node embeddings as a { model_name: float32 array } .npz archive.

    Args:
        mode      : "standard" or "newllm" — selects default graph/save paths
        graph     : path to the .pt graph file (None → auto from mode)
        K         : number of propagation hops (default 2)
        norm      : normalisation mode: sym | right | left | none (default sym)
        normalize : whether to L2-normalise output embeddings
        save      : output .npz path (None → auto from mode)
        keep      : model names to save; None → TARGET_MODELS; [] → all
    """
    import os
    ROOT_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    _PD       = os.path.join(ROOT_DIR, "profile_data")
    _RESULTS  = os.path.join(ROOT_DIR, "results")

    graph_path = graph or os.path.join(_RESULTS, "result_data_graph", mode, "task_graph_full.pt")
    _save_dir  = os.path.join(_RESULTS, "model_profile_result", mode)
    os.makedirs(_save_dir, exist_ok=True)
    save_path  = save or os.path.join(_save_dir, "emb_gnn.npz")

    if keep is None:
        keep_names = TARGET_MODELS
    elif len(keep) == 0:
        keep_names = None
    else:
        keep_names = keep

    # 1. Load graph
    print(f"── Step 1: Load graph from '{graph_path}' ────────────────")
    data = torch.load(graph_path, weights_only=False)
    print(data)

    # snapshot of raw features for comparison
    raw_feats = {ntype: data[ntype].x.float().clone() for ntype in data.node_types}

    # 2. Propagate
    print(f"\n── Step 2: Propagate  K={K}, norm='{norm}' ──────────────")
    prop_feats = propagate(data, K=K, norm=norm, normalize=normalize)

    # 3. Summary
    print_propagation_summary(raw_feats, prop_feats, K=K, norm=norm)

    # 4. Save model node embeddings
    print(f"\n── Step 3: Save model embeddings to '{save_path}' ───────")
    if keep_names is not None:
        print(f"  Filtering to {len(keep_names)} requested models ...")
    save_model_embeddings(prop_feats, data, save_path, keep_names=keep_names)

    print("\n✅ Done!")


def cli() -> None:
    import argparse
    import os
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    parser = argparse.ArgumentParser(description="Training-free heterogeneous GNN propagation.")
    parser.add_argument("--mode",      choices=["standard", "newllm"], default="standard",
                        help="Routing setting: standard or newllm (default: standard)")
    parser.add_argument("--graph",     default=None,
                        help="Input .pt graph file (default: profile_data/result_data_graph/{mode}/task_graph_full.pt)")
    parser.add_argument("--K",         default=2,   type=int,          help="Number of propagation hops")
    parser.add_argument("--norm",      default="sym",                  help="Normalisation: sym | right | left | none")
    parser.add_argument("--normalize", action="store_true",            help="L2-normalise output embeddings")
    parser.add_argument("--save",      default=None,
                        help="Output .npz file (default: routeprofile/model_profile_result/{mode}/emb_gnn.npz)")
    parser.add_argument(
        "--keep",
        nargs="*",
        default=None,
        metavar="MODEL",
        help="Model name keys to save. Omit to use TARGET_MODELS list; "
             "pass --keep with no arguments to save ALL models.",
    )
    args = parser.parse_args()

    build_model_profile(
        mode=args.mode,
        graph=args.graph,
        K=args.K,
        norm=args.norm,
        normalize=args.normalize,
        save=args.save,
        keep=args.keep,
    )


if __name__ == "__main__":
    cli()