"""
Build a PyTorch Geometric (PyG) heterogeneous graph from LLM model metadata.

Node types:
  - architecture : node feature text = architecture name  (e.g. "LlamaForCausalLM")
  - model        : node feature text = model "feature" description (long text)
  - query        : node feature text = query string

  All three node types share the same interface:
      .node_feature_text  – list[str]   raw text fed into Longformer
      .x                  – FloatTensor [N, D]  Longformer CLS embedding

Edge types (undirected, stored as both directions):
  - (architecture, arch_to_model,    model)
  - (model,        model_to_arch,    architecture)
  - (query,        query_to_model,   model)        ← NEW (via dataset bridge)
  - (model,        model_to_query,   query)         ← NEW

Usage:
  python build_llm_graph.py                                        # uses default paths
  python build_llm_graph.py data.json arch.json datasets.json out.pt  # custom paths

Install dependencies:
  pip install torch torch_geometric transformers
"""

import argparse
import json
import os
from pathlib import Path

import torch
from torch_geometric.data import HeteroData
from llmrouter.utils import get_longformer_embedding

# ── Project root (RouteProfile/) ──────────────────────────────────────────────
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PD = os.path.join(ROOT_DIR, "profile_data")

# ── Default paths (standard mode; override with --mode newllm) ────────────────
DEFAULT_JSON         = os.path.join(_PD, "model_feature_standard.json")
DEFAULT_ARCH_JSON    = os.path.join(_PD, "model_family_feature.json")
DEFAULT_DATASET_JSON = os.path.join(_PD, "task_feature.json")
DEFAULT_QUERY_JSON   = os.path.join(_PD, "task_queries_standard.json")
DEFAULT_SAVE_PT      = os.path.join(_PD, "result_data_graph", "standard", "query_graph_full.pt")


# ── 1. Data loading ────────────────────────────────────────────────────────────

def load_raw_data(json_path: str) -> dict:
    """Load raw LLM metadata from a JSON file."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    print(f"Loaded {len(raw)} model entries from '{path}'")
    return raw


def load_feature_texts(json_path: str) -> dict[str, str]:
    """Load a name→description mapping from a JSON file."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    print(f"Loaded {len(mapping)} feature entries from '{path}'")
    return mapping


def load_query_data(json_path: str) -> dict[str, list[str]]:
    """Load per-dataset query lists from all_dataset_queries.json."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Query JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    total = sum(len(v) for v in data.values())
    print(f"Loaded {total} queries across {len(data)} datasets from '{path}'")
    return data


# ── 2. Node collection ─────────────────────────────────────────────────────────

def collect_nodes(
    raw:           dict,
    arch_features: dict[str, str],
) -> tuple[
    list[str], list[str], dict[str, int],
    list[str], list[str], dict[str, int],
    dict[str, set[str]],  # dataset_name → set of model_keys with non-null score
]:
    """
    Parse raw data and collect unique nodes for architecture and model.

    Also returns dataset→model mapping (only non-null scores) for later
    use when building query→model edges.
    """
    arch_names:          list[str]      = []
    arch_feature_texts:  list[str]      = []
    arch_name2idx:       dict[str, int] = {}
    model_names:         list[str]      = []
    model_feature_texts: list[str]      = []
    model_name2idx:      dict[str, int] = {}

    # dataset → set of model keys that have a non-null score on this dataset
    # used later to bridge query→dataset→model into query→model edges
    dataset_to_models:   dict[str, set[str]] = {}

    for model_key, info in raw.items():
        # architecture node
        arch = info["architecture"]
        if arch not in arch_name2idx:
            if arch not in arch_features:
                raise KeyError(f"Architecture '{arch}' not found in arch_features JSON.")
            arch_name2idx[arch] = len(arch_names)
            arch_names.append(arch)
            arch_feature_texts.append(arch_features[arch])

        # model node
        model_name2idx[model_key] = len(model_names)
        model_names.append(model_key)
        model_feature_texts.append(info["feature"])

        # collect dataset→model mapping for non-null scores
        for ds, score in info["detailed_scores"].items():
            if score is None:
                continue
            dataset_to_models.setdefault(ds, set()).add(model_key)

    print(f"Unique architectures : {len(arch_names)}  → {arch_names}")
    print(f"Models               : {len(model_names)}")
    print(f"Datasets with scores : {len(dataset_to_models)}")

    return (
        arch_names, arch_feature_texts, arch_name2idx,
        model_names, model_feature_texts, model_name2idx,
        dataset_to_models,
    )


def collect_query_nodes(
    query_data:        dict[str, list[str]],
    dataset_to_models: dict[str, set[str]],
    model_name2idx:    dict[str, int],
) -> tuple[list[str], list[list[int]]]:
    """
    Collect query nodes and build query→model adjacency via dataset bridge.

    For each query belonging to dataset D, the query is connected to every
    model that has a non-null benchmark score on D.

    Returns:
        query_texts        : list[str]        – one entry per query node
        query_model_ids    : list[list[int]]  – for each query, list of model indices
    """
    query_texts:     list[str]       = []
    query_model_ids: list[list[int]] = []

    skipped_datasets = 0
    for ds_name, queries in query_data.items():
        if ds_name not in dataset_to_models:
            print(f"  Warning: dataset '{ds_name}' has no model scores — skipped.")
            skipped_datasets += 1
            continue
        model_indices = [model_name2idx[m] for m in dataset_to_models[ds_name]]
        for q in queries:
            query_texts.append(q)
            query_model_ids.append(model_indices)

    n_datasets = len(query_data) - skipped_datasets
    print(f"Query nodes : {len(query_texts)}  (across {n_datasets} datasets)")
    return query_texts, query_model_ids


# ── 3. Edge construction ───────────────────────────────────────────────────────

def build_edge_indices(
    raw:           dict,
    arch_name2idx: dict[str, int],
) -> dict[str, tuple[list[int], list[int]]]:
    """Build arch↔model edge index lists (dataset edges removed)."""
    arch_model_src, arch_model_dst = [], []
    model_arch_src, model_arch_dst = [], []

    for model_idx, (_, info) in enumerate(raw.items()):
        arch_idx = arch_name2idx[info["architecture"]]
        arch_model_src.append(arch_idx);  arch_model_dst.append(model_idx)
        model_arch_src.append(model_idx); model_arch_dst.append(arch_idx)

    return {
        "arch_to_model": (arch_model_src, arch_model_dst),
        "model_to_arch": (model_arch_src, model_arch_dst),
    }


def build_query_model_edge_indices(
    query_model_ids: list[list[int]],
) -> dict[str, tuple[list[int], list[int]]]:
    """
    Build query↔model edge index lists.

    Each query is connected to all models that have a non-null score on the
    dataset the query belongs to.
    """
    query_to_model_src, query_to_model_dst = [], []
    model_to_query_src, model_to_query_dst = [], []

    for query_idx, model_indices in enumerate(query_model_ids):
        for model_idx in model_indices:
            query_to_model_src.append(query_idx); query_to_model_dst.append(model_idx)
            model_to_query_src.append(model_idx); model_to_query_dst.append(query_idx)

    print(f"  query_to_model edges : {len(query_to_model_src)}")
    print(f"  model_to_query edges : {len(model_to_query_src)}")
    return {
        "query_to_model": (query_to_model_src, query_to_model_dst),
        "model_to_query": (model_to_query_src, model_to_query_dst),
    }


# ── 4. Longformer embedding ────────────────────────────────────────────────────

def encode_texts(texts: list[str], batch_size: int = 8) -> torch.Tensor:
    """
    Encode a list of strings into Longformer embeddings in small batches.

    Single-text batches (e.g. the last batch when len % batch_size == 1)
    may return a 1-D tensor [D]; these are unsqueezed to [1, D] before
    concatenation to avoid a RuntimeError.
    """
    all_embeddings: list[torch.Tensor] = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        emb   = get_longformer_embedding(batch)
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb, dtype=torch.float)
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)   # [D] → [1, D]
        all_embeddings.append(emb.cpu())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"  Encoded {min(i + batch_size, total)}/{total} texts")

    return torch.cat(all_embeddings, dim=0)   # [N, D]


def encode_all_nodes(
    arch_feature_texts:  list[str],
    model_feature_texts: list[str],
    query_texts:         list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode feature texts for all three node types using Longformer."""
    print("\n[1/3] Encoding architecture nodes ...")
    arch_emb = encode_texts(arch_feature_texts)

    print("\n[2/3] Encoding model nodes ...")
    model_emb = encode_texts(model_feature_texts)

    print("\n[3/3] Encoding query nodes ...")
    query_emb = encode_texts(query_texts)

    return arch_emb, model_emb, query_emb


# ── 5. Graph assembly ──────────────────────────────────────────────────────────

def assemble_graph(
    arch_names:          list[str],
    arch_feature_texts:  list[str],
    arch_emb:            torch.Tensor,
    model_names:         list[str],
    model_feature_texts: list[str],
    model_emb:           torch.Tensor,
    query_texts:         list[str],
    query_emb:           torch.Tensor,
    edge_indices:        dict[str, tuple[list[int], list[int]]],
    query_edge_indices:  dict[str, tuple[list[int], list[int]]],
) -> HeteroData:
    """Assemble all nodes and edges into a PyG HeteroData object."""
    data = HeteroData()

    # ── nodes ──
    data["architecture"].x                 = arch_emb
    data["architecture"].node_feature_text = arch_feature_texts
    data["architecture"].node_names        = arch_names

    data["model"].x                 = model_emb
    data["model"].node_feature_text = model_feature_texts
    data["model"].node_names        = model_names

    data["query"].x                 = query_emb
    data["query"].node_feature_text = query_texts

    def to_edge_index(src: list[int], dst: list[int]) -> torch.Tensor:
        return torch.tensor([src, dst], dtype=torch.long)

    # ── edges ──
    src, dst = edge_indices["arch_to_model"]
    data["architecture", "arch_to_model", "model"].edge_index = to_edge_index(src, dst)

    src, dst = edge_indices["model_to_arch"]
    data["model", "model_to_arch", "architecture"].edge_index = to_edge_index(src, dst)

    src, dst = query_edge_indices["query_to_model"]
    data["query", "query_to_model", "model"].edge_index = to_edge_index(src, dst)

    src, dst = query_edge_indices["model_to_query"]
    data["model", "model_to_query", "query"].edge_index = to_edge_index(src, dst)

    return data


# ── 6. Summary printing ────────────────────────────────────────────────────────

def print_summary(data: HeteroData) -> None:
    """Print a human-readable summary of the assembled graph."""
    print("\n" + "=" * 60)
    print("HeteroData Graph Summary")
    print("=" * 60)
    print(data)

    print("\n--- Node counts & embedding shape ---")
    for ntype in data.node_types:
        print(f"  {ntype:15s}: {data[ntype].num_nodes} nodes, "
              f"x.shape={list(data[ntype].x.shape)}")

    print("\n--- Edge counts ---")
    for edge_type in data.edge_types:
        n_edges = data[edge_type].edge_index.shape[1]
        print(f"  {str(edge_type):55s}: {n_edges} edges")

    print("\n--- Architecture nodes ---")
    for i, (name, feat) in enumerate(
        zip(data["architecture"].node_names, data["architecture"].node_feature_text)
    ):
        print(f"  [{i}] {name}")
        print(f"       {feat[:90]}...")

    print("\n--- Model nodes ---")
    for i, (name, feat) in enumerate(
        zip(data["model"].node_names, data["model"].node_feature_text)
    ):
        print(f"  [{i}] {name}")
        print(f"       {feat[:90]}...")

    print("\n--- Query nodes (first 5) ---")
    for i, feat in enumerate(data["query"].node_feature_text[:5]):
        print(f"  [{i}] {feat[:90]}...")
    if len(data["query"].node_feature_text) > 5:
        print(f"  ... ({len(data['query'].node_feature_text)} total)")


# ── 7. Save graph ──────────────────────────────────────────────────────────────

def save_graph(data: HeteroData, save_path: str) -> None:
    """Persist the HeteroData graph to disk as a .pt file."""
    torch.save(data, save_path)
    print(f"\n💾 Graph saved to: '{save_path}'")
    print(f"   Load with: data = torch.load('{save_path}', weights_only=False)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(
    json_path:    str = DEFAULT_JSON,
    arch_path:    str = DEFAULT_ARCH_JSON,
    dataset_path: str = DEFAULT_DATASET_JSON,
    query_path:   str = DEFAULT_QUERY_JSON,
    save_path:    str = DEFAULT_SAVE_PT,
) -> None:
    print("── Step 1: Load data ─────────────────────────────────────")
    raw           = load_raw_data(json_path)
    arch_features = load_feature_texts(arch_path)
    # dataset_feature JSON仍然需要加载，因为 collect_nodes 用它来确定哪些模型有分数
    # 但 dataset 节点本身不再进入图中
    dataset_features = load_feature_texts(dataset_path)
    query_data    = load_query_data(query_path)

    print("\n── Step 2: Collect nodes ─────────────────────────────────")
    (
        arch_names, arch_feature_texts, arch_name2idx,
        model_names, model_feature_texts, model_name2idx,
        dataset_to_models,
    ) = collect_nodes(raw, arch_features)

    query_texts, query_model_ids = collect_query_nodes(
        query_data, dataset_to_models, model_name2idx
    )

    print("\n── Step 3: Build edges ───────────────────────────────────")
    edge_indices       = build_edge_indices(raw, arch_name2idx)
    query_edge_indices = build_query_model_edge_indices(query_model_ids)
    print(f"  Arch/model edges : {list(edge_indices.keys())}")
    print(f"  Query/model edges: {list(query_edge_indices.keys())}")

    print("\n── Step 4: Encode node features (Longformer) ────────────")
    arch_emb, model_emb, query_emb = encode_all_nodes(
        arch_feature_texts, model_feature_texts, query_texts
    )

    print("\n── Step 5: Assemble graph ────────────────────────────────")
    data = assemble_graph(
        arch_names, arch_feature_texts, arch_emb,
        model_names, model_feature_texts, model_emb,
        query_texts, query_emb,
        edge_indices,
        query_edge_indices,
    )
    print("  Graph assembled.")

    print("\n── Step 6: Summary ───────────────────────────────────────")
    print_summary(data)

    print("\n── Step 7: Save graph ────────────────────────────────────")
    save_graph(data, save_path)

    print("\n✅ Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build query graph (architecture/model/query) for RouteProfile."
    )
    parser.add_argument("--mode",    choices=["standard", "newllm"], default="standard",
                        help="Routing setting: standard or newllm (default: standard)")
    parser.add_argument("--json",    default=None, metavar="PATH",
                        help="Override model feature JSON path")
    parser.add_argument("--arch",    default=None, metavar="PATH",
                        help="Override architecture feature JSON path")
    parser.add_argument("--dataset", default=None, metavar="PATH",
                        help="Override dataset feature JSON path")
    parser.add_argument("--query",   default=None, metavar="PATH",
                        help="Override query JSON path")
    parser.add_argument("--save",    default=None, metavar="PATH",
                        help="Override output .pt path")
    _args = parser.parse_args()

    _out_dir      = os.path.join(_PD, "result_data_graph", _args.mode)
    os.makedirs(_out_dir, exist_ok=True)

    _json_path    = _args.json    or os.path.join(_PD, f"model_feature_{_args.mode}.json")
    _arch_path    = _args.arch    or os.path.join(_PD, "model_family_feature.json")
    _dataset_path = _args.dataset or os.path.join(_PD, "task_feature.json")
    _query_path   = _args.query   or os.path.join(_PD, f"task_queries_{_args.mode}.json")
    _save_path    = _args.save    or os.path.join(_out_dir, "query_graph_full.pt")

    main(_json_path, _arch_path, _dataset_path, _query_path, _save_path)