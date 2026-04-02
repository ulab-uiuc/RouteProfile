"""
Build a PyTorch Geometric (PyG) heterogeneous graph from LLM model metadata.

Node types:
  - architecture : node feature text = architecture name  (e.g. "LlamaForCausalLM")
  - model        : node feature text = model "feature" description (long text)
  - dataset      : node feature text = dataset name  (e.g. "ifeval")

  All three node types share the same interface:
      .node_feature_text  – list[str]   raw text fed into BERT
      .x                  – FloatTensor [N, 768]  BERT CLS embedding

Edge types (undirected, stored as both directions):
  - (architecture, arch_to_model,    model)
  - (model,        model_to_arch,    architecture)
  - (model,        model_to_dataset, dataset)
  - (dataset,      dataset_to_model, model)

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
DEFAULT_SAVE_PT      = os.path.join(_PD, "result_data_graph", "standard", "task_graph_full.pt")


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
    """
    Load a name→description mapping from a JSON file.
 
    Used for architecture and dataset node feature texts.
    Expected format: { "NodeName": "description text", ... }
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    print(f"Loaded {len(mapping)} feature entries from '{path}'")
    return mapping
 
 
# ── 2. Node collection ─────────────────────────────────────────────────────────
 
def collect_nodes(
    raw:              dict,
    arch_features:    dict[str, str],
    dataset_features: dict[str, str],
) -> tuple[
    list[str], list[str], dict[str, int],
    list[str], list[str],
    list[str], list[str], dict[str, int],
]:
    """
    Parse raw data and collect unique nodes for each node type.
 
    Dataset nodes are only registered when at least one model has a non-null
    score for them — datasets where every model returns null are excluded
    entirely from the graph.
 
    Args:
        raw:              Raw LLM metadata loaded from JSON.
        arch_features:    Mapping of architecture name → description text.
        dataset_features: Mapping of dataset name → description text.
 
    Returns:
        arch_names, arch_feature_texts, arch_name2idx
        model_names, model_feature_texts
        dataset_names, dataset_feature_texts, dataset_name2idx
    """
    arch_names:           list[str]      = []
    arch_feature_texts:   list[str]      = []
    arch_name2idx:        dict[str, int] = {}
    model_names:          list[str]      = []
    model_feature_texts:  list[str]      = []
    dataset_names:        list[str]      = []
    dataset_feature_texts: list[str]    = []
    dataset_name2idx:     dict[str, int] = {}
 
    for model_key, info in raw.items():
        # architecture node — skip if unknown (masked out)
        arch = info["architecture"]
        if arch != "unknown":
            if arch not in arch_name2idx:
                if arch not in arch_features:
                    raise KeyError(f"Architecture '{arch}' not found in arch_features JSON.")
                arch_name2idx[arch] = len(arch_names)
                arch_names.append(arch)
                arch_feature_texts.append(arch_features[arch])
 
        # dataset nodes — only register when score is non-null.
        # A dataset node is created the first time any model has a real score for it.
        for ds, score in info["detailed_scores"].items():
            if score is None:
                continue                          # null score → skip
            if ds not in dataset_name2idx:
                if ds not in dataset_features:
                    raise KeyError(f"Dataset '{ds}' not found in dataset_features JSON.")
                dataset_name2idx[ds] = len(dataset_names)
                dataset_names.append(ds)
                dataset_feature_texts.append(dataset_features[ds])
 
        # model node
        model_names.append(model_key)
        model_feature_texts.append(info["feature"])
 
    print(f"Unique architectures : {len(arch_names)}  → {arch_names}")
    print(f"Unique datasets      : {len(dataset_names)}  → {dataset_names}")
    print(f"Models               : {len(model_names)}")
 
    return (
        arch_names, arch_feature_texts, arch_name2idx,
        model_names, model_feature_texts,
        dataset_names, dataset_feature_texts, dataset_name2idx,
    )
 
 
# ── 3. Edge construction ───────────────────────────────────────────────────────
 
def build_edge_indices(
    raw:              dict,
    arch_name2idx:    dict[str, int],
    dataset_name2idx: dict[str, int],
) -> dict[str, tuple[list[int], list[int], list[float] | None]]:
    """
    Build edge index lists (and edge attributes where applicable) for all edge types.
 
    Edge attributes:
      - model_to_dataset / dataset_to_model : the benchmark score (float scalar)
        stored as a [E, 1] tensor.
      - arch_to_model / model_to_arch       : no edge attributes (None)
 
    Null scores are skipped — no edge is created for that (model, dataset) pair.
 
    Returns:
        Dict keyed by relation name; each value is (src_list, dst_list, scores_or_None).
    """
    arch_model_src,    arch_model_dst    = [], []
    model_arch_src,    model_arch_dst    = [], []
    model_dataset_src, model_dataset_dst = [], []
    dataset_model_src, dataset_model_dst = [], []
 
    model_dataset_scores: list[float] = []
    dataset_model_scores: list[float] = []
 
    n_skipped = 0
 
    for model_idx, (_, info) in enumerate(raw.items()):
        arch = info["architecture"]
 
        # architecture <-> model  (no edge feature)
        # skip when arch is unknown — no arch node exists for this model
        if arch != "unknown":
            arch_idx = arch_name2idx[arch]
            arch_model_src.append(arch_idx);  arch_model_dst.append(model_idx)
            model_arch_src.append(model_idx); model_arch_dst.append(arch_idx)
 
        # model <-> dataset  (edge feature = benchmark score, null → no edge)
        for ds, score in info["detailed_scores"].items():
            if score is None:
                n_skipped += 1
                continue
            if ds not in dataset_name2idx:
                # dataset never registered (all models scored null for it)
                continue
            ds_idx = dataset_name2idx[ds]
            model_dataset_src.append(model_idx); model_dataset_dst.append(ds_idx)
            model_dataset_scores.append(float(score))
 
            dataset_model_src.append(ds_idx);    dataset_model_dst.append(model_idx)
            dataset_model_scores.append(float(score))
 
    print(f"  Skipped {n_skipped} null score entries (no edge created).")
 
    return {
        "arch_to_model":    (arch_model_src,    arch_model_dst,    None),
        "model_to_arch":    (model_arch_src,    model_arch_dst,    None),
        "model_to_dataset": (model_dataset_src, model_dataset_dst, model_dataset_scores),
        "dataset_to_model": (dataset_model_src, dataset_model_dst, dataset_model_scores),
    }
 
 
# ── 4. Longformer embedding ────────────────────────────────────────────────────
 
def encode_texts(texts: list[str], batch_size: int = 8) -> torch.Tensor:
    """
    Encode a list of strings into Longformer embeddings via
    get_longformer_embedding(), processing in small batches to avoid OOM.
 
    Args:
        texts:      List of input strings.
        batch_size: Number of texts per forward pass (default 8).
 
    Returns:
        FloatTensor of shape [N, D].
    """
    all_embeddings: list[torch.Tensor] = []
    total = len(texts)
 
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        emb   = get_longformer_embedding(batch)
        if not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb, dtype=torch.float)
        # ensure 2D: single-text batches may return shape [D] instead of [1, D]
        if emb.dim() == 1:
            emb = emb.unsqueeze(0)
        all_embeddings.append(emb.cpu())
 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
 
        print(f"  Encoded {min(i + batch_size, total)}/{total} texts")
 
    return torch.cat(all_embeddings, dim=0)   # [N, D]
 
 
def encode_all_nodes(
    arch_feature_texts:    list[str],
    model_feature_texts:   list[str],
    dataset_feature_texts: list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Encode feature texts for all three node types using Longformer.
 
    Returns:
        arch_emb, model_emb, dataset_emb  – each FloatTensor [N, D]
    """
    print("\n[1/3] Encoding architecture nodes ...")
    arch_emb = encode_texts(arch_feature_texts)
 
    print("\n[2/3] Encoding model nodes ...")
    model_emb = encode_texts(model_feature_texts)
 
    print("\n[3/3] Encoding dataset nodes ...")
    dataset_emb = encode_texts(dataset_feature_texts)
 
    return arch_emb, model_emb, dataset_emb
 
 
# ── 5. Graph assembly ──────────────────────────────────────────────────────────
 
def assemble_graph(
    arch_names:            list[str],
    arch_feature_texts:    list[str],
    arch_emb:              torch.Tensor,
    model_names:           list[str],
    model_feature_texts:   list[str],
    model_emb:             torch.Tensor,
    dataset_names:         list[str],
    dataset_feature_texts: list[str],
    dataset_emb:           torch.Tensor,
    edge_indices:          dict[str, tuple[list[int], list[int], list[float] | None]],
) -> HeteroData:
    """
    Assemble all nodes and edges into a PyG HeteroData object.
 
    Returns:
        A fully populated HeteroData graph.
    """
    data = HeteroData()
 
    # nodes ── architecture
    data["architecture"].x                 = arch_emb
    data["architecture"].node_feature_text = arch_feature_texts
    data["architecture"].node_names        = arch_names
 
    # nodes ── model
    data["model"].x                 = model_emb
    data["model"].node_feature_text = model_feature_texts
    data["model"].node_names        = model_names
 
    # nodes ── dataset
    data["dataset"].x                 = dataset_emb
    data["dataset"].node_feature_text = dataset_feature_texts
    data["dataset"].node_names        = dataset_names
 
    # edges
    def to_edge_index(src: list[int], dst: list[int]) -> torch.Tensor:
        return torch.tensor([src, dst], dtype=torch.long)
 
    def to_edge_attr(scores: list[float] | None) -> torch.Tensor | None:
        if scores is None:
            return None
        return torch.tensor(scores, dtype=torch.float).unsqueeze(1)  # [E, 1]
 
    src, dst, scores = edge_indices["arch_to_model"]
    data["architecture", "arch_to_model", "model"].edge_index = to_edge_index(src, dst)
 
    src, dst, scores = edge_indices["model_to_arch"]
    data["model", "model_to_arch", "architecture"].edge_index = to_edge_index(src, dst)
 
    src, dst, scores = edge_indices["model_to_dataset"]
    data["model", "model_to_dataset", "dataset"].edge_index = to_edge_index(src, dst)
    data["model", "model_to_dataset", "dataset"].edge_attr  = to_edge_attr(scores)
 
    src, dst, scores = edge_indices["dataset_to_model"]
    data["dataset", "dataset_to_model", "model"].edge_index = to_edge_index(src, dst)
    data["dataset", "dataset_to_model", "model"].edge_attr  = to_edge_attr(scores)
 
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
 
    print("\n--- Edge counts & attributes ---")
    for edge_type in data.edge_types:
        es       = data[edge_type]
        n_edges  = es.edge_index.shape[1]
        has_attr = hasattr(es, "edge_attr") and es.edge_attr is not None
        attr_info = f"  edge_attr.shape={list(es.edge_attr.shape)}" if has_attr else ""
        print(f"  {str(edge_type):55s}: {n_edges} edges{attr_info}")
 
    print("\n--- Architecture nodes ---")
    for i, (name, feat) in enumerate(
        zip(data["architecture"].node_names, data["architecture"].node_feature_text)
    ):
        print(f"  [{i}] {name}")
        print(f"       {feat[:90]}...")
 
    print("\n--- Dataset nodes ---")
    for i, (name, feat) in enumerate(
        zip(data["dataset"].node_names, data["dataset"].node_feature_text)
    ):
        print(f"  [{i}] {name}")
        print(f"       {feat[:90]}...")
 
    print("\n--- Model nodes ---")
    for i, (name, feat) in enumerate(
        zip(data["model"].node_names, data["model"].node_feature_text)
    ):
        print(f"  [{i}] {name}")
        print(f"       {feat[:90]}...")
 
 
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
    save_path:    str = DEFAULT_SAVE_PT,
) -> None:
    # 1. Load raw data and feature texts from JSON files
    print("── Step 1: Load data ─────────────────────────────────────")
    raw              = load_raw_data(json_path)
    arch_features    = load_feature_texts(arch_path)
    dataset_features = load_feature_texts(dataset_path)
 
    # 2. Collect unique nodes
    print("\n── Step 2: Collect nodes ─────────────────────────────────")
    (
        arch_names, arch_feature_texts, arch_name2idx,
        model_names, model_feature_texts,
        dataset_names, dataset_feature_texts, dataset_name2idx,
    ) = collect_nodes(raw, arch_features, dataset_features)
 
    # 3. Build edge indices
    print("\n── Step 3: Build edges ───────────────────────────────────")
    edge_indices = build_edge_indices(raw, arch_name2idx, dataset_name2idx)
    print(f"  Edge relations built: {list(edge_indices.keys())}")
 
    # 4. Encode node features with Longformer
    print("\n── Step 4: Encode node features (Longformer) ────────────")
    arch_emb, model_emb, dataset_emb = encode_all_nodes(
        arch_feature_texts, model_feature_texts, dataset_feature_texts
    )
 
    # 5. Assemble HeteroData graph
    print("\n── Step 5: Assemble graph ────────────────────────────────")
    data = assemble_graph(
        arch_names, arch_feature_texts, arch_emb,
        model_names, model_feature_texts, model_emb,
        dataset_names, dataset_feature_texts, dataset_emb,
        edge_indices,
    )
    print("  Graph assembled.")
 
    # 6. Print summary
    print("\n── Step 6: Summary ───────────────────────────────────────")
    print_summary(data)
 
    # 7. Save to disk
    print("\n── Step 7: Save graph ────────────────────────────────────")
    save_graph(data, save_path)
 
    print("\n✅ Done!")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build task graph (architecture/model/dataset) for RouteProfile."
    )
    parser.add_argument("--mode",    choices=["standard", "newllm"], default="standard",
                        help="Routing setting: standard or newllm (default: standard)")
    parser.add_argument("--json",    default=None, metavar="PATH",
                        help="Override model feature JSON path")
    parser.add_argument("--arch",    default=None, metavar="PATH",
                        help="Override architecture feature JSON path")
    parser.add_argument("--dataset", default=None, metavar="PATH",
                        help="Override dataset feature JSON path")
    parser.add_argument("--save",    default=None, metavar="PATH",
                        help="Override output .pt path")
    _args = parser.parse_args()

    _out_dir      = os.path.join(_PD, "result_data_graph", _args.mode)
    os.makedirs(_out_dir, exist_ok=True)

    _json_path    = _args.json    or os.path.join(_PD, f"model_feature_{_args.mode}.json")
    _arch_path    = _args.arch    or os.path.join(_PD, "model_family_feature.json")
    _dataset_path = _args.dataset or os.path.join(_PD, "task_feature.json")
    _save_path    = _args.save    or os.path.join(_out_dir, "task_graph_full.pt")

    main(_json_path, _arch_path, _dataset_path, _save_path)