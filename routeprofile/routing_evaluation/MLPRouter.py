"""
train_mlp_router.py
--------------------
MLP-based router trained with pairwise ranking loss.

Architecture
------------
  query_mlp  : Longformer(query) → MLP → query_emb   [out_dim]
  model_mlp  : model_profile     → MLP → model_emb   [out_dim]
  score(q,m) = cosine_sim(query_emb, model_emb)
  selected   = argmax score over all models

Both MLPs share the same architecture but have separate weights.

Training
--------
  Data   : pairwise_training_data_sampled.json
  Loss   : max(0, margin - sim(q, better) + sim(q, worse))
  Query embeddings are computed once (frozen Longformer) and cached.
  Model profile embeddings are loaded from a pre-computed .npz file (fixed).

Evaluation
----------
  Same metrics as EmbeddingSimRouter / GraphRouter:
    task_performance, is_hit, is_discriminative,
    avg_performance, total_avg_hit, performance_summary
  Test data: routing_test_data.json

Usage
-----
    python train_mlp_router.py \\
        --profiles    /path/to/model_profiles.npz \\
        --train-data  /path/to/pairwise_training_data_sampled.json \\
        --test-data   /path/to/routing_test_data.json \\
        --output      /path/to/mlp_router_results.json \\
        --save-ckpt   /path/to/mlp_router_ckpt.pt \\
        --cache       /path/to/train_query_emb.pt
"""

import argparse
import ast
import copy
import json
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from llmrouter.utils import get_longformer_embedding, generate_task_query, calculate_task_performance


# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_HIDDEN_DIM  = 512
DEFAULT_OUT_DIM     = 256
DEFAULT_NUM_LAYERS  = 2
DEFAULT_DROPOUT     = 0.1
DEFAULT_MARGIN      = 0.2
DEFAULT_LR          = 1e-4
DEFAULT_EPOCHS      = 20
DEFAULT_BATCH_SIZE  = 32
DEFAULT_SEED        = 42
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


# ── Helpers ────────────────────────────────────────────────────────────────────

def _format_query_text(original_query: str, task_name: Optional[str], row: dict) -> str:
    if task_name:
        try:
            raw_choices = row.get("choices", None)
            if isinstance(raw_choices, str):
                choices_list = ast.literal_eval(raw_choices)
            elif isinstance(raw_choices, list):
                choices_list = raw_choices
            else:
                choices_list = None
            sample_data = {"query": original_query, "choices": choices_list}
            formatted   = generate_task_query(task_name, sample_data)
            user_text   = formatted.get("user") or ""
            system_text = formatted.get("system") or ""
            combined    = (system_text + "\n\n" + user_text).strip() if system_text else user_text.strip()
            return combined if combined else original_query
        except Exception as e:
            print(f"  Warning: query formatting failed ({e}). Using raw query.")
    return original_query


# ── MLP ───────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    """
    Feed-forward MLP: in_dim → hidden_dim → ... → out_dim.
    num_layers hidden layers, each with LayerNorm + ELU + Dropout.
    Output is L2-normalised.
    """

    def __init__(
        self,
        in_dim:     int,
        hidden_dim: int,
        out_dim:    int,
        num_layers: int = 2,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")

        layers: list[nn.Module] = []
        for i in range(num_layers):
            in_ch  = in_dim    if i == 0 else hidden_dim
            out_ch = out_dim   if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_ch, out_ch))
            if i < num_layers - 1:
                layers.append(nn.LayerNorm(out_ch))
                layers.append(nn.ELU())
                layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)


class MLPRouter(nn.Module):
    """
    Two separate MLPs — one for queries, one for model profiles.
    score(q, m) = dot(query_mlp(q_emb), model_mlp(m_emb))
    """

    def __init__(
        self,
        in_dim:     int,
        hidden_dim: int,
        out_dim:    int,
        num_layers: int,
        dropout:    float,
    ) -> None:
        super().__init__()
        self.query_mlp = MLP(in_dim, hidden_dim, out_dim, num_layers, dropout)
        self.model_mlp = MLP(in_dim, hidden_dim, out_dim, num_layers, dropout)

    def encode_query(self, q: Tensor) -> Tensor:
        return self.query_mlp(q)             # [B, out_dim]

    def encode_models(self, m: Tensor) -> Tensor:
        return self.model_mlp(m)             # [N_models, out_dim]

    def score(self, q_emb: Tensor, m_emb: Tensor) -> Tensor:
        """cosine sim: q_emb [B, D] × m_emb [N_m, D] → [B, N_m]"""
        return q_emb @ m_emb.T


# ── Pairwise dataset ──────────────────────────────────────────────────────────

class PairwiseDataset(torch.utils.data.Dataset):
    """
    Wraps pairwise_training_data_sampled.json.
    Items: { query_emb: Tensor[768], better_name: str, worse_name: str }
    """

    def __init__(
        self,
        json_path:   str,
        model_names: list[str],
        device:      torch.device,
        cache_path:  Optional[str] = None,
    ) -> None:
        print(f"Loading pairwise data from '{json_path}' ...")
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        pairs = raw.get("pairwise_data", raw) if isinstance(raw, dict) else raw

        model_name_set = set(model_names)
        valid_pairs    = [
            p for p in pairs
            if p["better_model"] in model_name_set
            and p["worse_model"]  in model_name_set
        ]
        print(f"  {len(valid_pairs)} / {len(pairs)} pairs kept.")

        self.pairs      = valid_pairs
        self.device     = device
        self.query_embs: list[Tensor] = []

        if cache_path and os.path.exists(cache_path):
            print(f"  Loading cached query embeddings from '{cache_path}' ...")
            cached = torch.load(cache_path, map_location="cpu")
            self.query_embs = [e.to(device) for e in cached]
            print(f"  {len(self.query_embs)} embeddings loaded.")
        else:
            print("  Computing query embeddings (Longformer, frozen) ...")
            for i, p in enumerate(valid_pairs):
                text = _format_query_text(p["query"], p.get("task_name"), p)
                emb  = get_longformer_embedding(text)
                if isinstance(emb, np.ndarray):
                    emb = torch.from_numpy(emb)
                self.query_embs.append(emb.float().squeeze().to(device))
                if (i + 1) % 200 == 0:
                    print(f"    {i + 1}/{len(valid_pairs)} done.")
            print(f"  Done. {len(self.query_embs)} embeddings computed.")
            if cache_path:
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                torch.save([e.cpu() for e in self.query_embs], cache_path)
                print(f"  Cached to '{cache_path}'.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        p = self.pairs[idx]
        return {
            "query_emb":   self.query_embs[idx],
            "better_name": p["better_model"],
            "worse_name":  p["worse_model"],
        }


def _collate(batch: list[dict]) -> dict:
    return {
        "query_emb":   torch.stack([b["query_emb"]   for b in batch]),
        "better_name": [b["better_name"] for b in batch],
        "worse_name":  [b["worse_name"]  for b in batch],
    }


# ── Evaluation (mirrors EmbeddingSimRouter / GraphRouter) ─────────────────────

def _evaluate(
    test_rows:      list[dict],
    selected_names: list[str],
) -> dict:
    routing_results   = []
    perf_summary: dict[str, list[float]] = {}
    hit_summary:  dict[str, list[bool]]  = {}

    for row, model_name in zip(test_rows, selected_names):
        row_copy   = copy.copy(row)
        model_perf = row_copy.get("model_performance", {})
        task_name  = row_copy.get("task_name", "unknown")

        row_copy["model_name"] = model_name
        selected = model_perf.get(model_name, {})
        row_copy["response"]          = selected.get("response", "")
        row_copy["success"]           = selected.get("success", False)
        row_copy["prompt_tokens"]     = selected.get("prompt_tokens", 0)
        row_copy["completion_tokens"] = selected.get("completion_tokens", 0)

        if "task_performance" in selected and selected["task_performance"] is not None:
            row_copy["task_performance"] = selected["task_performance"]
        else:
            gt = (row_copy.get("ground_truth") or row_copy.get("gt")
                  or row_copy.get("answer"))
            if gt and row_copy["success"]:
                row_copy["task_performance"] = calculate_task_performance(
                    prediction=row_copy["response"],
                    ground_truth=gt,
                    task_name=task_name,
                    metric=row_copy.get("metric"),
                )
            else:
                row_copy["task_performance"] = None

        all_perfs = [
            v.get("task_performance")
            for v in model_perf.values()
            if v.get("task_performance") is not None
        ]
        has_zero = any(p == 0 for p in all_perfs)
        has_one  = any(p == 1 for p in all_perfs)
        row_copy["is_discriminative"] = bool(has_zero and has_one)

        all_zero      = all_perfs and all(p == 0 for p in all_perfs)
        selected_perf = row_copy.get("task_performance")
        if all_zero or not all_perfs:
            row_copy["is_hit"] = True
        else:
            row_copy["is_hit"] = bool(selected_perf is not None and selected_perf > 0)

        row_copy.pop("model_performance", None)
        routing_results.append(row_copy)

        if row_copy["task_performance"] is not None:
            perf_summary.setdefault(task_name, []).append(row_copy["task_performance"])
        if row_copy.get("is_hit") is not None:
            hit_summary.setdefault(task_name, []).append(row_copy["is_hit"])

    task_perf_sum: dict[str, float] = {}
    hit_perf_sum:  dict[str, float] = {}
    for task, perfs in perf_summary.items():
        avg     = sum(perfs) / len(perfs)
        hits    = hit_summary.get(task, [])
        avg_hit = sum(hits) / len(hits) if hits else None
        task_perf_sum[task] = avg
        if avg_hit is not None:
            hit_perf_sum[task] = avg_hit
        hit_str = f"  avg_hit={avg_hit:.4f}" if avg_hit is not None else ""
        print(f"  📊 {task}: {avg:.4f}  (n={len(perfs)}){hit_str}")

    all_perfs_flat = [p for ps in perf_summary.values() for p in ps]
    all_hits_flat  = [h for hs in hit_summary.values()  for h in hs]
    avg_perf = sum(all_perfs_flat) / len(all_perfs_flat) if all_perfs_flat else None
    avg_hit  = sum(all_hits_flat)  / len(all_hits_flat)  if all_hits_flat  else None
    if avg_perf is not None:
        print(f"  📊 avg_performance : {avg_perf:.4f}  (n={len(all_perfs_flat)})")
    if avg_hit is not None:
        print(f"  📊 total_avg_hit   : {avg_hit:.4f}  (n={len(all_hits_flat)})")

    return {
        "avg_performance": avg_perf,
        "total_avg_hit":   avg_hit,
        "performance_summary": {
            "task_performance": task_perf_sum,
            "hit_performance":  hit_perf_sum,
        },
        "routing_results": routing_results,
    }


# ── Training + evaluation ──────────────────────────────────────────────────────

def train(
    profiles_path:    str,
    train_data_path:  str,
    test_data_path:   str,
    output_path:      str,
    save_ckpt_path:   str,
    cache_path:       Optional[str],
    hidden_dim:       int,
    out_dim:          int,
    num_layers:       int,
    dropout:          float,
    margin:           float,
    lr:               float,
    epochs:           int,
    batch_size:       int,
    seed:             int,
    val_split:        float,
) -> None:

    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. load model profile embeddings ─────────────────────────────────────
    print(f"\n── Step 1: Load model profiles from '{profiles_path}' ───")
    loaded        = np.load(profiles_path)
    model_names   = list(loaded.files)
    profile_np    = np.stack([loaded[n] for n in model_names])   # [N, 768]
    profile_emb   = torch.from_numpy(profile_np).float().to(device)
    name2idx      = {n: i for i, n in enumerate(model_names)}
    in_dim        = profile_emb.size(1)
    print(f"  {len(model_names)} models, in_dim={in_dim}")

    # ── 2. build MLPRouter ────────────────────────────────────────────────────
    print(f"\n── Step 2: Build MLPRouter  "
          f"(hidden={hidden_dim}, out={out_dim}, layers={num_layers}) ──")
    router = MLPRouter(
        in_dim=in_dim, hidden_dim=hidden_dim, out_dim=out_dim,
        num_layers=num_layers, dropout=dropout,
    ).to(device)
    total_p = sum(p.numel() for p in router.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_p:,}")

    optimizer = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.1
    )

    # ── 3. pairwise dataset ───────────────────────────────────────────────────
    print(f"\n── Step 3: Load pairwise data from '{train_data_path}' ──")
    dataset = PairwiseDataset(
        json_path=train_data_path,
        model_names=model_names,
        device=device,
        cache_path=cache_path,
    )
    n_total = len(dataset)
    n_val   = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    indices = list(range(n_total))
    random.shuffle(indices)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices[:n_train]),
        batch_size=batch_size, shuffle=True, collate_fn=_collate,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, indices[n_train:]),
        batch_size=batch_size, shuffle=False, collate_fn=_collate,
    )
    print(f"  Train: {n_train}  |  Val: {n_val}")

    # ── 4. training loop ──────────────────────────────────────────────────────
    print(f"\n── Step 4: Training  (epochs={epochs}, margin={margin}) ──")
    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(1, epochs + 1):

        # ── train ─────────────────────────────────────────────────────────────
        router.train()
        train_loss, n_b = 0.0, 0

        for batch in train_loader:
            q_raw        = batch["query_emb"].to(device)     # [B, 768]
            better_names = batch["better_name"]
            worse_names  = batch["worse_name"]

            # encode models fresh each batch so the computation graph is intact
            m_emb = router.encode_models(profile_emb)        # [N_models, out_dim]
            q_emb = router.encode_query(q_raw)               # [B, out_dim]

            better_idx = torch.tensor([name2idx[n] for n in better_names], device=device)
            worse_idx  = torch.tensor([name2idx[n] for n in worse_names],  device=device)

            sim_better = (q_emb * m_emb[better_idx]).sum(dim=1)   # [B]
            sim_worse  = (q_emb * m_emb[worse_idx ]).sum(dim=1)   # [B]
            loss       = F.relu(margin - sim_better + sim_worse).mean()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_b        += 1

        avg_train = train_loss / max(n_b, 1)

        # ── validate ──────────────────────────────────────────────────────────
        router.eval()
        val_loss, n_correct, n_val_total = 0.0, 0, 0
        with torch.no_grad():
            m_emb_val = router.encode_models(profile_emb)
            for batch in val_loader:
                q_raw        = batch["query_emb"].to(device)
                better_names = batch["better_name"]
                worse_names  = batch["worse_name"]

                q_emb = router.encode_query(q_raw)

                better_idx = torch.tensor([name2idx[n] for n in better_names], device=device)
                worse_idx  = torch.tensor([name2idx[n] for n in worse_names],  device=device)

                sim_better = (q_emb * m_emb_val[better_idx]).sum(dim=1)
                sim_worse  = (q_emb * m_emb_val[worse_idx ]).sum(dim=1)
                loss       = F.relu(margin - sim_better + sim_worse).mean()

                val_loss   += loss.item()
                n_correct  += (sim_better > sim_worse).sum().item()
                n_val_total += len(better_names)

        avg_val = val_loss / max(len(val_loader), 1)
        val_acc = n_correct / max(n_val_total, 1)
        scheduler.step()

        print(f"  Epoch {epoch:3d}/{epochs}  "
              f"train={avg_train:.4f}  val={avg_val:.4f}  "
              f"val_acc={val_acc:.4f}  lr={optimizer.param_groups[0]['lr']:.2e}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state    = {k: v.cpu().clone() for k, v in router.state_dict().items()}
            os.makedirs(os.path.dirname(save_ckpt_path) or ".", exist_ok=True)
            torch.save({
                "epoch": epoch, "model_state": best_state,
                "val_loss": avg_val, "val_acc": val_acc,
                "config": {
                    "in_dim": in_dim, "hidden_dim": hidden_dim,
                    "out_dim": out_dim, "num_layers": num_layers,
                    "model_names": model_names,
                },
            }, save_ckpt_path)
            print(f"    ✅ Best checkpoint saved (val_loss={avg_val:.4f})")

    print(f"\n  Best val_loss: {best_val_loss:.4f}")

    # ── 5. test inference ─────────────────────────────────────────────────────
    print(f"\n── Step 5: Test inference on '{test_data_path}' ─────────")
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_rows: list[dict] = json.load(f)
    print(f"  {len(test_rows)} test queries.")

    router.load_state_dict(best_state)
    router.eval()

    selected_names: list[str] = []
    with torch.no_grad():
        m_emb_final = router.encode_models(profile_emb)   # [N_models, out_dim]

        for row in test_rows:
            text  = _format_query_text(row["query"], row.get("task_name"), row)
            q_raw = get_longformer_embedding(text)
            if isinstance(q_raw, np.ndarray):
                q_raw = torch.from_numpy(q_raw)
            q_raw = q_raw.float().squeeze().unsqueeze(0).to(device)  # [1, 768]
            q_emb = router.encode_query(q_raw)                        # [1, out_dim]
            scores = router.score(q_emb, m_emb_final).squeeze(0)     # [N_models]
            selected_names.append(model_names[scores.argmax().item()])

    # ── 6. evaluate + save ────────────────────────────────────────────────────
    print(f"\n── Step 6: Evaluation ────────────────────────────────────")
    results = _evaluate(test_rows, selected_names)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to '{output_path}'")
    print(f"✅ Done!  Best val_loss={best_val_loss:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import os
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    _PR = os.path.join(ROOT_DIR, "routeprofile")
    _RD = os.path.join(ROOT_DIR, "route_data")

    parser = argparse.ArgumentParser(
        description="MLPRouter: pairwise ranking loss, test on routing_test_data."
    )
    parser.add_argument("--mode",        choices=["standard", "newllm"], default="standard",
                        help="Routing setting: standard or newllm (default: standard)")
    parser.add_argument("--profiles",    default=None,
                        help="Path to model profile embeddings (.npz) "
                             "(default: routeprofile/model_profile_result/{mode}/flat.npz)")
    parser.add_argument("--train-data",  default=None,
                        help="Path to routing_train_data.json "
                             "(default: route_data/routing_train_data.json)")
    parser.add_argument("--test-data",   default=None,
                        help="Path to routing_test_data.json "
                             "(default: route_data/routing_test_data.json)")
    parser.add_argument("--output",      default=None,
                        help="Output results JSON "
                             "(default: routeprofile/routing_result/{mode}/MLPRouter_results.json)")
    parser.add_argument("--save-ckpt",   default=None,
                        help="Checkpoint path "
                             "(default: routeprofile/routing_evaluation/trained_MLPRouter/{mode}/mlp_router_ckpt.pt)")
    parser.add_argument("--cache",       default=None,
                        help="Cache path for train query embeddings (.pt)")
    parser.add_argument("--hidden-dim",  type=int,   default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--out-dim",     type=int,   default=DEFAULT_OUT_DIM)
    parser.add_argument("--num-layers",  type=int,   default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--dropout",     type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--margin",      type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--lr",          type=float, default=DEFAULT_LR)
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size",  type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed",        type=int,   default=DEFAULT_SEED)
    parser.add_argument("--val-split",   type=float, default=0.1)
    args = parser.parse_args()

    _result_dir = os.path.join(_PR, "routing_result", args.mode)
    _ckpt_dir   = os.path.join(_PR, "routing_evaluation", "trained_MLPRouter", args.mode)
    os.makedirs(_result_dir, exist_ok=True)
    os.makedirs(_ckpt_dir,   exist_ok=True)

    train(
        profiles_path=args.profiles   or os.path.join(_PR, "model_profile_result", args.mode, "flat.npz"),
        train_data_path=args.train_data or os.path.join(_RD, "routing_train_data.json"),
        test_data_path=args.test_data  or os.path.join(_RD, "routing_test_data.json"),
        output_path=args.output        or os.path.join(_result_dir, "MLPRouter_results.json"),
        save_ckpt_path=args.save_ckpt  or os.path.join(_ckpt_dir, "mlp_router_ckpt.pt"),
        cache_path=args.cache,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        margin=args.margin,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        val_split=args.val_split,
    )


if __name__ == "__main__":
    main()