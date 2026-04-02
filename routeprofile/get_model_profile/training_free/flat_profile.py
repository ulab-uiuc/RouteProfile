"""
plain_text_embedding_random.py
-------------------------------
Build model profile embeddings by concatenating the model's own
node_feature_text with RANDOMLY sampled nodes from every other node type,
then encoding the concatenated text with Longformer.

Unlike the similarity-based version, neighbours are chosen uniformly at
random (per node type) rather than by cosine similarity to the model.

Usage
-----
    python plain_text_embedding_random.py \\
        --graph  /path/to/hetero_graph.pt \\
        --save   /path/to/random_text_embeddings.npz \\
        --top-k  5 \\
        --seed   42
"""

import argparse
import os
import random

import numpy as np
import torch

from torch_geometric.data import HeteroData
from llmrouter.utils import get_longformer_embedding


# ── Constants ─────────────────────────────────────────────────────────────────
DEFAULT_TOP_K = 5
DEFAULT_SEED  = 42
DEFAULT_KEEP_MODELS = [
    "qwen2.5-7b-instruct",
    "gemma-2-9b-it",
    "llama-3.1-8b-instruct",
    "mixtral-8x7b-instruct-v0.1",
    "mixtral-8x22b-instruct-v0.1",
    "llama-3.2-3b-instruct",
    "mistral-small-24b-instruct-2501-bf16",
    "llama-3.3-70b-instruct",
]


# ── Longformer encoding ───────────────────────────────────────────────────────

def encode_texts(texts: list[str], batch_size: int = 8) -> np.ndarray:
    """Encode texts with Longformer; returns L2-normalised [N, D] float32."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        emb   = get_longformer_embedding(batch)
        if isinstance(emb, torch.Tensor):
            if emb.dim() == 1:
                emb = emb.unsqueeze(0)
            emb = emb.cpu().numpy()
        emb   = np.atleast_2d(emb).astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb   = emb / np.where(norms < 1e-8, 1.0, norms)
        all_embs.append(emb)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"  Encoded {min(i + batch_size, len(texts))}/{len(texts)}")
    return np.concatenate(all_embs, axis=0)


# ── Core function ─────────────────────────────────────────────────────────────

def random_text_embed(
    data:        HeteroData,
    top_k:       int = DEFAULT_TOP_K,
    seed:        int = DEFAULT_SEED,
    keep_names:  list[str] | None = None,
    batch_size:  int = 8,
) -> dict[str, np.ndarray]:
    """
    For each model node:
      1. Take the model's own node_feature_text.
      2. From every other node type, randomly sample up to top_k nodes.
      3. Concatenate all texts (model first, then sampled nodes by type).
      4. Encode the concatenated string with Longformer → [768] embedding.

    Args:
        data        : HeteroData graph with .node_feature_text on all node types
        top_k       : number of nodes to randomly sample per other node type
        seed        : random seed for reproducibility
        keep_names  : model names to include; None = all models in the graph
        batch_size  : Longformer batch size

    Returns:
        dict { model_name: np.ndarray [768] }
    """
    rng = random.Random(seed)

    model_names: list[str] = list(data["model"].node_names)
    other_types: list[str] = [t for t in data.node_types if t != "model"]

    if keep_names is not None:
        keep_set       = set(keep_names)
        target_indices = [i for i, n in enumerate(model_names) if n in keep_set]
        target_names   = [model_names[i] for i in target_indices]
    else:
        target_indices = list(range(len(model_names)))
        target_names   = model_names

    # collect all node texts per type (other than model)
    other_texts: dict[str, list[str]] = {}
    for ntype in other_types:
        if hasattr(data[ntype], "node_feature_text"):
            other_texts[ntype] = list(data[ntype].node_feature_text)
        else:
            print(f"  Warning: node type '{ntype}' has no node_feature_text — skipping.")

    print(f"Building random-sampled profiles for {len(target_names)} models  "
          f"(top_k={top_k} per node type, seed={seed}) ...")

    concatenated_texts: list[str] = []

    for model_idx in target_indices:
        model_name = model_names[model_idx]
        model_text = data["model"].node_feature_text[model_idx]

        parts: list[str] = [f"[MODEL] {model_text}"]

        for ntype, texts in other_texts.items():
            k        = min(top_k, len(texts))
            sampled  = rng.sample(texts, k)
            for text in sampled:
                parts.append(f"[{ntype.upper()}] {text}")

        concatenated = "\n".join(parts)
        concatenated_texts.append(concatenated)
        print(f"  [{model_name}] {len(parts)} segments, {len(concatenated)} chars")

    print(f"\nEncoding {len(concatenated_texts)} concatenated texts ...")
    emb_matrix = encode_texts(concatenated_texts, batch_size=batch_size)

    return {name: emb_matrix[i] for i, name in enumerate(target_names)}


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import os
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    _PD = os.path.join(ROOT_DIR, "profile_data")
    _PR = os.path.join(ROOT_DIR, "routeprofile")

    parser = argparse.ArgumentParser(
        description="Random-sampled plain-text model profile embeddings."
    )
    parser.add_argument("--mode",       choices=["standard", "newllm"], default="standard",
                        help="Routing setting: standard or newllm (default: standard)")
    parser.add_argument("--graph",      default=None,
                        help="Path to the HeteroData .pt graph file "
                             "(default: profile_data/result_data_graph/{mode}/task_graph_full.pt)")
    parser.add_argument("--save",       default=None,
                        help="Output .npz path "
                             "(default: routeprofile/model_profile_result/{mode}/flat.npz)")
    parser.add_argument("--top-k",      type=int, default=DEFAULT_TOP_K,
                        help=f"Nodes to randomly sample per node type (default {DEFAULT_TOP_K})")
    parser.add_argument("--seed",       type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default {DEFAULT_SEED})")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--keep",       nargs="*", default=None, metavar="MODEL",
                        help="Model names to include. "
                             "Omit = DEFAULT_KEEP_MODELS; --keep with no args = all.")
    args = parser.parse_args()

    _graph_default = os.path.join(_PD, "result_data_graph", args.mode, "task_graph_full.pt")
    _save_dir = os.path.join(_PR, "model_profile_result", args.mode)
    os.makedirs(_save_dir, exist_ok=True)
    _save_default = os.path.join(_save_dir, "flat.npz")

    if args.keep is None:
        keep = DEFAULT_KEEP_MODELS
    elif len(args.keep) == 0:
        keep = None
    else:
        keep = args.keep

    graph_path = args.graph or _graph_default
    save_path  = args.save  or _save_default

    print(f"Loading graph from '{graph_path}' ...")
    data = torch.load(graph_path, weights_only=False)
    print(data)

    embeddings = random_text_embed(
        data=data,
        top_k=args.top_k,
        seed=args.seed,
        keep_names=keep,
        batch_size=args.batch_size,
    )

    np.savez(save_path, **embeddings)
    print(f"\n✅ Saved {len(embeddings)} embeddings → '{save_path}'")


if __name__ == "__main__":
    main()