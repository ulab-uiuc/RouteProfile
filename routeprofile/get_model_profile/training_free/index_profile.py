"""
index_profile.py
-----------------
Random-vector baseline: assigns each candidate model a random unit vector.
Useful as a lower-bound baseline for routing experiments.

Usage:
    python index_profile.py
    python index_profile.py --mode newllm --save /custom/path/random.npz
"""

import argparse
import os

import numpy as np
import torch

ROOT_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_RESULTS  = os.path.join(ROOT_DIR, "results")

EMBEDDING_DIM = 768
DEFAULT_SEED  = 56

TARGET_MODELS: list[str] = [
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
    mode: str       = "standard",
    save: str | None = None,
    seed: int       = DEFAULT_SEED,
) -> None:
    """
    Generate random unit-vector baseline profiles for all candidate models.

    Args:
        mode : "standard" or "newllm" — selects default save path
        save : output .npz path (None → auto from mode)
        seed : random seed (default 56)
    """
    _save_dir = os.path.join(_RESULTS, "model_profile_result", mode)
    os.makedirs(_save_dir, exist_ok=True)
    save_path = save or os.path.join(_save_dir, "index.npz")

    torch.manual_seed(seed)
    model_vecs = {
        name: torch.randn(EMBEDDING_DIM, dtype=torch.float32)
        for name in TARGET_MODELS
    }

    np.savez(save_path, **{k: v.numpy() for k, v in model_vecs.items()})
    print(f"✅ Saved {len(model_vecs)} random embeddings (dim={EMBEDDING_DIM}) → '{save_path}'")


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Random-vector index profile (baseline for LLM routing)."
    )
    parser.add_argument("--mode", choices=["standard", "newllm"], default="standard",
                        help="Routing setting: standard or newllm (default: standard)")
    parser.add_argument("--save", default=None, metavar="PATH",
                        help="Output .npz path "
                             "(default: routeprofile/model_profile_result/{mode}/index.npz)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Random seed (default {DEFAULT_SEED})")
    args = parser.parse_args()
    build_model_profile(mode=args.mode, save=args.save, seed=args.seed)


if __name__ == "__main__":
    cli()