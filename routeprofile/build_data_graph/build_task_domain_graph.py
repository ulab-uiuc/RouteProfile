"""
Build a PyTorch Geometric (PyG) heterogeneous graph from LLM model metadata.

Node types:
  - architecture : node feature text = architecture description
  - model        : node feature text = model "feature" description
  - dataset      : node feature text = dataset description

  All three node types share the same interface:
      .node_feature_text  – list[str]   raw text fed into Longformer
      .x                  – FloatTensor [N, D]  Longformer embedding

Edge types (undirected, stored as both directions):
  - (architecture, arch_to_model,    model)
  - (model,        model_to_arch,    architecture)
  - (model,        model_to_dataset, dataset)   ← only when score is not null
  - (dataset,      dataset_to_model, model)     ← only when score is not null

  model <-> dataset edges carry the benchmark score as edge_attr [E, 1].
  Entries where detailed_scores[dataset] is null are silently skipped:
  no edge is created and the dataset node is only registered if at least
  one other model has a non-null score for it.

Usage:
  python build_llm_graph.py                                           # default paths
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
ROOT_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PD       = os.path.join(ROOT_DIR, "profile_data")
_RESULTS  = os.path.join(ROOT_DIR, "results")

# ── Default paths (standard mode; override with --mode newllm) ────────────────
DEFAULT_DOMAIN_MAP_JSON  = os.path.join(_PD, "domain_task_map.json")
DEFAULT_DOMAIN_FEAT_JSON = os.path.join(_PD, "domain_feature.json")
DEFAULT_JSON             = os.path.join(_PD, "model_feature_standard.json")
DEFAULT_ARCH_JSON        = os.path.join(_PD, "model_family_feature.json")
DEFAULT_DATASET_JSON     = os.path.join(_PD, "task_feature.json")
DEFAULT_SAVE_PT          = os.path.join(_RESULTS, "result_data_graph", "standard", "task_domain_graph_full.pt")


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


def load_domain_data(
    map_path:  str,
    feat_path: str,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """
    Load domain → dataset mapping and domain feature texts.

    Args:
        map_path  : path to domain_dataset_map.json  { domain: [dataset, ...] }
        feat_path : path to domain_feature.json      { domain: description }

    Returns:
        domain_map      : { domain_name: [dataset_name, ...] }
        domain_features : { domain_name: description_text }
    """
    for path in (map_path, feat_path):
        if not Path(path).exists():
            raise FileNotFoundError(f"Domain file not found: {path}")
    with Path(map_path).open("r", encoding="utf-8") as f:
        domain_map: dict[str, list[str]] = json.load(f)
    with Path(feat_path).open("r", encoding="utf-8") as f:
        domain_features: dict[str, str] = json.load(f)
    print(f"Loaded {len(domain_map)} domains from '{map_path}'")
    return domain_map, domain_features


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
        # architecture node
        arch = info["architecture"]
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


def collect_domain_nodes(
    domain_map:       dict[str, list[str]],
    domain_features:  dict[str, str],
    dataset_name2idx: dict[str, int],
) -> tuple[list[str], list[str], dict[str, int], list[int], list[int]]:
    """
    Collect domain nodes and domain↔dataset edge lists.

    Only edges to datasets present in the graph are created.
    """
    domain_names:         list[str]      = []
    domain_feature_texts: list[str]      = []
    domain_name2idx:      dict[str, int] = {}
    domain_dataset_src:   list[int]      = []
    domain_dataset_dst:   list[int]      = []

    for domain_name, datasets in domain_map.items():
        if domain_name not in domain_features:
            raise KeyError(f"Domain '{domain_name}' not found in domain_feature.json.")
        domain_idx = len(domain_names)
        domain_name2idx[domain_name] = domain_idx
        domain_names.append(domain_name)
        domain_feature_texts.append(domain_features[domain_name])
        for ds_name in datasets:
            if ds_name not in dataset_name2idx:
                continue    # dataset absent from graph → skip edge
            domain_dataset_src.append(domain_idx)
            domain_dataset_dst.append(dataset_name2idx[ds_name])

    print(f"Domain nodes : {len(domain_names)}  → {domain_names}")
    print(f"Domain↔dataset edges : {len(domain_dataset_src)} (x2 with reverse)")
    return (
        domain_names, domain_feature_texts, domain_name2idx,
        domain_dataset_src, domain_dataset_dst,
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
        arch_idx = arch_name2idx[info["architecture"]]

        # architecture <-> model  (no edge feature)
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
    domain_feature_texts:  list[str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Encode feature texts for all four node types using Longformer.

    Returns:
        arch_emb, model_emb, dataset_emb, domain_emb  – each FloatTensor [N, D]
    """
    print("\n[1/4] Encoding architecture nodes ...")
    arch_emb = encode_texts(arch_feature_texts)

    print("\n[2/4] Encoding model nodes ...")
    model_emb = encode_texts(model_feature_texts)

    print("\n[3/4] Encoding dataset nodes ...")
    dataset_emb = encode_texts(dataset_feature_texts)

    print("\n[4/4] Encoding domain nodes ...")
    domain_emb = encode_texts(domain_feature_texts)

    return arch_emb, model_emb, dataset_emb, domain_emb


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
    domain_names:          list[str],
    domain_feature_texts:  list[str],
    domain_emb:            torch.Tensor,
    domain_dataset_src:    list[int],
    domain_dataset_dst:    list[int],
    edge_indices:          dict[str, tuple[list[int], list[int], list[float] | None]],
) -> HeteroData:
    """
    Assemble all nodes and edges into a PyG HeteroData object.

    Node types : architecture, model, dataset, domain
    Edge types :
      architecture <-> model   (structural, no edge attr)
      model        <-> dataset (benchmark score as edge attr)
      domain       <-> dataset (membership, no edge attr)

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

    # nodes ── domain
    data["domain"].x                 = domain_emb
    data["domain"].node_feature_text = domain_feature_texts
    data["domain"].node_names        = domain_names

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

    # domain <-> dataset  (undirected, no edge attr)
    data["domain",  "domain_to_dataset", "dataset"].edge_index = to_edge_index(domain_dataset_src, domain_dataset_dst)
    data["dataset", "dataset_to_domain", "domain"].edge_index  = to_edge_index(domain_dataset_dst, domain_dataset_src)

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

    print("\n--- Domain nodes ---")
    for i, (name, feat) in enumerate(
        zip(data["domain"].node_names, data["domain"].node_feature_text)
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
    json_path:        str = DEFAULT_JSON,
    arch_path:        str = DEFAULT_ARCH_JSON,
    dataset_path:     str = DEFAULT_DATASET_JSON,
    domain_map_path:  str = DEFAULT_DOMAIN_MAP_JSON,
    domain_feat_path: str = DEFAULT_DOMAIN_FEAT_JSON,
    save_path:        str = DEFAULT_SAVE_PT,
) -> None:
    # 1. Load raw data and feature texts from JSON files
    print("── Step 1: Load data ─────────────────────────────────────")
    raw              = load_raw_data(json_path)
    arch_features    = load_feature_texts(arch_path)
    dataset_features = load_feature_texts(dataset_path)
    domain_map, domain_features = load_domain_data(domain_map_path, domain_feat_path)

    # 2. Collect unique nodes
    print("\n── Step 2: Collect nodes ─────────────────────────────────")
    (
        arch_names, arch_feature_texts, arch_name2idx,
        model_names, model_feature_texts,
        dataset_names, dataset_feature_texts, dataset_name2idx,
    ) = collect_nodes(raw, arch_features, dataset_features)

    (
        domain_names, domain_feature_texts, domain_name2idx,
        domain_dataset_src, domain_dataset_dst,
    ) = collect_domain_nodes(domain_map, domain_features, dataset_name2idx)

    # 3. Build edge indices
    print("\n── Step 3: Build edges ───────────────────────────────────")
    edge_indices = build_edge_indices(raw, arch_name2idx, dataset_name2idx)
    print(f"  Edge relations built: {list(edge_indices.keys())}")

    # 4. Encode node features with Longformer
    print("\n── Step 4: Encode node features (Longformer) ────────────")
    arch_emb, model_emb, dataset_emb, domain_emb = encode_all_nodes(
        arch_feature_texts, model_feature_texts,
        dataset_feature_texts, domain_feature_texts,
    )

    # 5. Assemble HeteroData graph
    print("\n── Step 5: Assemble graph ────────────────────────────────")
    data = assemble_graph(
        arch_names, arch_feature_texts, arch_emb,
        model_names, model_feature_texts, model_emb,
        dataset_names, dataset_feature_texts, dataset_emb,
        domain_names, domain_feature_texts, domain_emb,
        domain_dataset_src, domain_dataset_dst,
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


def build_task_domain_graph(
    mode:        str        = "standard",
    json:        str | None = None,
    arch:        str | None = None,
    dataset:     str | None = None,
    domain_map:  str | None = None,
    domain_feat: str | None = None,
    save:        str | None = None,
) -> None:
    """
    Build the task-domain graph (architecture / model / dataset / domain nodes).

    Args:
        mode        : "standard" or "newllm"
        json        : path to model feature JSON (None → auto)
        arch        : path to architecture feature JSON (None → shared default)
        dataset     : path to dataset feature JSON (None → shared default)
        domain_map  : path to domain→dataset map JSON (None → shared default)
        domain_feat : path to domain feature JSON (None → shared default)
        save        : output .pt path (None → auto)
    """
    _out_dir = os.path.join(_RESULTS, "result_data_graph", mode)
    os.makedirs(_out_dir, exist_ok=True)
    main(
        json_path=json          or os.path.join(_PD, f"model_feature_{mode}.json"),
        arch_path=arch          or os.path.join(_PD, "model_family_feature.json"),
        dataset_path=dataset    or os.path.join(_PD, "task_feature.json"),
        domain_map_path=domain_map  or os.path.join(_PD, "domain_task_map.json"),
        domain_feat_path=domain_feat or os.path.join(_PD, "domain_feature.json"),
        save_path=save          or os.path.join(_out_dir, "task_domain_graph_full.pt"),
    )


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Build task-domain graph (architecture/model/dataset/domain) for RouteProfile."
    )
    parser.add_argument("--mode",        choices=["standard", "newllm"], default="standard",
                        help="Routing setting: standard or newllm (default: standard)")
    parser.add_argument("--json",        default=None, metavar="PATH",
                        help="Override model feature JSON path")
    parser.add_argument("--arch",        default=None, metavar="PATH",
                        help="Override architecture feature JSON path")
    parser.add_argument("--dataset",     default=None, metavar="PATH",
                        help="Override dataset feature JSON path")
    parser.add_argument("--domain-map",  default=None, metavar="PATH",
                        help="Override domain→dataset map JSON path")
    parser.add_argument("--domain-feat", default=None, metavar="PATH",
                        help="Override domain feature JSON path")
    parser.add_argument("--save",        default=None, metavar="PATH",
                        help="Override output .pt path")
    _args = parser.parse_args()
    build_task_domain_graph(mode=_args.mode, json=_args.json, arch=_args.arch,
                            dataset=_args.dataset, domain_map=_args.domain_map,
                            domain_feat=_args.domain_feat, save=_args.save)


if __name__ == "__main__":
    cli()