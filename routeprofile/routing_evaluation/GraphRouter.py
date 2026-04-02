"""
train_graphrouter.py
---------------------
GraphRouter: bipartite query-model graph + GAT + edge prediction.

Graph structure
---------------
  Nodes:
    query : one per train/test query  (.x = Longformer embedding)
    model : one per LLM               (.x = model profile from .npz)

  Edges (fully-connected bipartite):
    train query ↔ model : edge_attr = task_performance  (known)
    test  query ↔ model : edge_attr = 0                 (unknown, to predict)

Training
--------
  Only discriminative train queries participate in the loss.
  A query is discriminative if at least one model scores 0 AND
  at least one model scores 1 (i.e. models are not all tied).

  For each discriminative train query q:
    label(q, m) = 1  if m achieves the highest task_performance for q
    label(q, m) = 0  otherwise   (handles ties: all tied-best get label=1)
  loss = BCE over all (q, m) edge pairs in the batch

  edge_predictor(concat(h_q, h_m)) → scalar logit per (q, m) pair

Inference
---------
  For each test query q:
    score(q, m) = sigmoid( edge_predictor(concat(h_q, h_m)) )
    selected    = argmax score over all models

Evaluation
----------
  Same metrics as EmbeddingSimRouter:
    task_performance, is_hit, is_discriminative,
    avg_performance, total_avg_hit, performance_summary

Usage
-----
    python train_graphrouter.py \\
        --profiles    /path/to/model_profiles.npz \\
        --train-data  /path/to/routing_train_data.json \\
        --test-data   /path/to/routing_test_data.json \\
        --output      /path/to/graphrouter_results.json \\
        --save-ckpt   /path/to/graphrouter_ckpt.pt \\
        --cache-train /path/to/train_q_emb.npy \\
        --cache-test  /path/to/test_q_emb.npy
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
from torch_geometric.nn import GATConv

from llmrouter.utils import get_longformer_embedding, generate_task_query, calculate_task_performance


# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_HIDDEN_DIM  = 256
DEFAULT_NUM_LAYERS  = 2
DEFAULT_HEADS       = 4
DEFAULT_DROPOUT     = 0.1
DEFAULT_LR          = 1e-4
DEFAULT_EPOCHS      = 100
DEFAULT_BATCH_SIZE  = 32
DEFAULT_SEED        = 42


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


def _encode_texts(texts: list[str], batch_size: int = 8) -> np.ndarray:
    """Encode texts; returns L2-normalised [N, D] float32."""
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
        print(f"    Encoded {min(i + batch_size, len(texts))}/{len(texts)}")
    return np.concatenate(all_embs, axis=0)


def _is_discriminative(perf_dict: dict[str, float]) -> bool:
    """True if at least one model scores 0 AND at least one scores 1."""
    vals = list(perf_dict.values())
    return any(v == 0 for v in vals) and any(v == 1 for v in vals)


# ── GAT encoder ───────────────────────────────────────────────────────────────

class BipGATEncoder(nn.Module):
    """
    Multi-layer GAT on the bipartite query-model graph.

    The graph is treated as homogeneous for simplicity (query and model nodes
    share the same feature dimension after projection).  GATConv is applied
    on the combined node set; query and model nodes are distinguished by their
    position in the node tensor.

    Architecture:
        project query + model features to hidden_dim
        → L × GATConv(hidden_dim, hidden_dim, heads)
        → output: h_query [N_q, hidden_dim], h_model [N_m, hidden_dim]
    """

    def __init__(
        self,
        q_in_dim:      int,
        m_in_dim:      int,
        hidden_dim:    int,
        num_layers:    int,
        heads:         int,
        dropout:       float,
        align_model:   bool = False,   # if True, add a linear to map model feat → q space
    ) -> None:
        super().__init__()
        # optional alignment: project model features into query feature space
        # before the shared hidden_dim projection
        if align_model and m_in_dim != q_in_dim:
            self.m_align = nn.Linear(m_in_dim, q_in_dim)
        else:
            self.m_align = None

        self.q_proj = nn.Linear(q_in_dim, hidden_dim)
        self.m_proj = nn.Linear(q_in_dim if (align_model and m_in_dim != q_in_dim)
                                else m_in_dim, hidden_dim)

        # each GAT layer outputs hidden_dim (heads are averaged, not concatenated)
        self.convs = nn.ModuleList([
            GATConv(hidden_dim, hidden_dim, heads=heads,
                    dropout=dropout, concat=False)
            for _ in range(num_layers)
        ])
        self.dropout    = nn.Dropout(dropout)
        self.num_layers = num_layers

    def forward(
        self,
        h_q_raw:    Tensor,    # [N_q, in_dim]
        h_m_raw:    Tensor,    # [N_m, in_dim]
        edge_index: Tensor,    # [2, E]  src/dst in combined node space
    ) -> tuple[Tensor, Tensor]:
        """
        Returns updated (h_query [N_q, H], h_model [N_m, H]).
        """
        N_q = h_q_raw.size(0)

        # optionally align model features to query feature space first
        h_m_in = self.m_align(h_m_raw) if self.m_align is not None else h_m_raw

        # project to hidden_dim
        h_q = F.elu(self.q_proj(h_q_raw))   # [N_q, H]
        h_m = F.elu(self.m_proj(h_m_in))    # [N_m, H]

        # concatenate into combined node tensor
        h = torch.cat([h_q, h_m], dim=0)    # [N_q + N_m, H]

        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index)
            h     = F.elu(h_new) if i < self.num_layers - 1 else h_new
            h     = self.dropout(h)

        return h[:N_q], h[N_q:]             # h_query, h_model


# ── Edge predictor ─────────────────────────────────────────────────────────────

class EdgePredictor(nn.Module):
    """
    MLP that predicts a scalar score for each (query, model) edge.
    Input: concat(h_query, h_model)  [2H]
    Output: scalar logit
    """

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h_q: Tensor,   # [B, H]
        h_m: Tensor,   # [N_m, H]
    ) -> Tensor:
        """Returns logits [B, N_m]."""
        B, H = h_q.shape
        N_m  = h_m.size(0)
        h_q_exp = h_q.unsqueeze(1).expand(B, N_m, H)
        h_m_exp = h_m.unsqueeze(0).expand(B, N_m, H)
        return self.net(torch.cat([h_q_exp, h_m_exp], dim=2)).squeeze(2)


# ── Full model ─────────────────────────────────────────────────────────────────

class GraphRouterModel(nn.Module):
    def __init__(
        self,
        q_in_dim:    int,
        m_in_dim:    int,
        hidden_dim:  int,
        num_layers:  int,
        heads:       int,
        dropout:     float,
        align_model: bool = False,
    ) -> None:
        super().__init__()
        self.encoder   = BipGATEncoder(
            q_in_dim=q_in_dim, m_in_dim=m_in_dim,
            hidden_dim=hidden_dim, num_layers=num_layers,
            heads=heads, dropout=dropout, align_model=align_model,
        )
        self.edge_pred = EdgePredictor(hidden_dim, dropout)

    def forward(
        self,
        h_q_raw:    Tensor,
        h_m_raw:    Tensor,
        edge_index: Tensor,
        batch_q_local_idx: Tensor,  # [B] indices into h_q (after encode)
    ) -> Tensor:
        """Returns logits [B, N_m]."""
        h_q, h_m = self.encoder(h_q_raw, h_m_raw, edge_index)
        return self.edge_pred(h_q[batch_q_local_idx], h_m)

    def encode(
        self,
        h_q_raw:    Tensor,
        h_m_raw:    Tensor,
        edge_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return self.encoder(h_q_raw, h_m_raw, edge_index)


# ── Graph construction ─────────────────────────────────────────────────────────

def build_graph(
    train_rows:    list[dict],
    test_rows:     list[dict],
    model_names:   list[str],
    model_name2idx: dict[str, int],
    profile_emb:   Tensor,          # [N_m, D] on CPU
    cache_train:   Optional[str],
    cache_test:    Optional[str],
) -> tuple[Tensor, Tensor, Tensor, list[dict], list[dict]]:
    """
    Build bipartite query-model graph.

    Node layout in combined tensor:
        [0 .. N_q_train-1]              → train query nodes
        [N_q_train .. N_q_train+N_q_test-1] → test query nodes
        (model nodes are kept separate and concatenated in forward)

    Returns:
        h_q_all    : [N_q_train + N_q_test, D]  query node features
        h_m        : [N_m, D]                    model node features (= profile_emb)
        edge_index : [2, E]  bipartite edges in combined (query+model) node space
                     src = query idx (0-based in h_q_all)
                     dst = model idx (0-based in h_m)
                     edges stored as directed both ways for GAT
        train_meta : [{ local_q_idx, perf_per_model, row }]
        test_meta  : [{ local_q_idx, row }]
    """
    N_m = len(model_names)

    # ── encode query nodes ────────────────────────────────────────────────────
    def _load_or_encode(rows: list[dict], cache: Optional[str], tag: str) -> np.ndarray:
        texts = [_format_query_text(r["query"], r.get("task_name"), r) for r in rows]
        if cache and os.path.exists(cache):
            print(f"  [{tag}] Loading cached embeddings from '{cache}' ...")
            return np.load(cache)
        print(f"  [{tag}] Encoding {len(texts)} query nodes ...")
        emb = _encode_texts(texts)
        if cache:
            os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
            np.save(cache, emb)
            print(f"  [{tag}] Cached to '{cache}'.")
        return emb

    train_emb = _load_or_encode(train_rows, cache_train, "train")  # [N_train, D]
    test_emb  = _load_or_encode(test_rows,  cache_test,  "test")   # [N_test,  D]

    N_q_train = len(train_rows)
    N_q_test  = len(test_rows)

    h_q_all = torch.from_numpy(
        np.concatenate([train_emb, test_emb], axis=0)
    ).float()  # [N_train + N_test, D]

    # ── build fully-connected bipartite edges ─────────────────────────────────
    # For GAT on combined node tensor [h_q_all; h_m]:
    #   query node i has global idx i
    #   model node j has global idx N_q_train + N_q_test + j
    # We store edges as undirected (both directions) so each node can attend
    # to its neighbours.
    N_q_total = N_q_train + N_q_test

    q_indices = torch.arange(N_q_total).repeat_interleave(N_m)   # [N_q*N_m]
    m_indices = torch.arange(N_q_total, N_q_total + N_m).repeat(N_q_total)  # [N_q*N_m]

    # undirected: q→m and m→q
    src = torch.cat([q_indices, m_indices])
    dst = torch.cat([m_indices, q_indices])
    edge_index = torch.stack([src, dst], dim=0)  # [2, 2*N_q*N_m]

    # ── metadata ──────────────────────────────────────────────────────────────
    train_meta: list[dict] = []
    for li, row in enumerate(train_rows):
        perf_per_model: dict[str, float] = {}
        for mname, mdata in row.get("model_performance", {}).items():
            if mname not in model_name2idx:
                continue
            perf = mdata.get("task_performance")
            if perf is None:
                continue
            perf_per_model[mname] = float(perf)
        train_meta.append({
            "local_q_idx":  li,   # index into h_q_all (= h_q_train part)
            "perf_per_model": perf_per_model,
            "row": row,
        })

    test_meta: list[dict] = []
    for li, row in enumerate(test_rows):
        test_meta.append({
            "local_q_idx": N_q_train + li,   # offset into h_q_all
            "row": row,
        })

    print(f"\n  Graph summary:")
    print(f"    query nodes : {N_q_total}  ({N_q_train} train + {N_q_test} test)")
    print(f"    model nodes : {N_m}")
    print(f"    edges       : {edge_index.size(1)}  (undirected, fully connected)")

    return h_q_all, profile_emb, edge_index, train_meta, test_meta


# ── Evaluation ─────────────────────────────────────────────────────────────────

def _evaluate(test_meta: list[dict], selected_names: list[str]) -> dict:
    routing_results   = []
    perf_summary: dict[str, list[float]] = {}
    hit_summary:  dict[str, list[bool]]  = {}

    for meta, model_name in zip(test_meta, selected_names):
        row        = meta["row"]
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


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    profiles_path:    str,
    train_data_path:  str,
    test_data_path:   str,
    output_path:      str,
    save_ckpt_path:   str,
    cache_train:      Optional[str],
    cache_test:       Optional[str],
    hidden_dim:       int,
    num_layers:       int,
    heads:            int,
    dropout:          float,
    lr:               float,
    epochs:           int,
    batch_size:       int,
    seed:             int,
    val_split:        float,
    align_model:      bool = False,
) -> None:

    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. load model profiles ────────────────────────────────────────────────
    print(f"\n── Step 1: Load model profiles from '{profiles_path}' ───")
    loaded       = np.load(profiles_path)
    model_names  = list(loaded.files)
    profile_np   = np.stack([loaded[n] for n in model_names])
    profile_emb  = torch.from_numpy(profile_np).float()   # [N_m, D] on CPU
    model_name2idx = {n: i for i, n in enumerate(model_names)}
    in_dim         = profile_emb.size(1)
    N_m            = len(model_names)
    print(f"  {N_m} models, in_dim={in_dim}")

    # ── 2. load routing data ──────────────────────────────────────────────────
    print(f"\n── Step 2: Load routing data ─────────────────────────────")
    with open(train_data_path, "r", encoding="utf-8") as f:
        train_rows: list[dict] = json.load(f)
    with open(test_data_path,  "r", encoding="utf-8") as f:
        test_rows:  list[dict] = json.load(f)
    print(f"  Raw train rows : {len(train_rows)}")
    print(f"  Test rows      : {len(test_rows)}")

    # ── 3. build graph ────────────────────────────────────────────────────────
    print(f"\n── Step 3: Build bipartite query-model graph ─────────────")
    h_q_all, h_m_cpu, edge_index, train_meta, test_meta = build_graph(
        train_rows=train_rows,
        test_rows=test_rows,
        model_names=model_names,
        model_name2idx=model_name2idx,
        profile_emb=profile_emb,
        cache_train=cache_train,
        cache_test=cache_test,
    )

    # move to device
    h_q_all    = h_q_all.to(device)
    h_m        = h_m_cpu.to(device)
    edge_index = edge_index.to(device)

    # ── 4. filter discriminative train queries ────────────────────────────────
    print(f"\n── Step 4: Filter discriminative train queries ───────────")
    disc_meta = [m for m in train_meta if _is_discriminative(m["perf_per_model"])]
    n_all     = len(train_meta)
    n_disc    = len(disc_meta)
    print(f"  Discriminative : {n_disc} / {n_all}  ({n_disc/max(n_all,1)*100:.1f}%)")

    # label distribution
    from collections import Counter
    best_cnt: Counter = Counter()
    for m in disc_meta:
        pd = m["perf_per_model"]
        bs = max(pd.values())
        for mn, p in pd.items():
            if p == bs:
                best_cnt[mn] += 1
    print("  Best-model distribution:")
    for mn, cnt in best_cnt.most_common():
        print(f"    {mn:<45s}: {cnt:4d}  ({cnt/n_disc*100:.1f}%)")

    # train / val split
    n_total = len(disc_meta)
    n_val   = max(1, int(n_total * val_split))
    n_train = n_total - n_val
    indices = list(range(n_total))
    random.shuffle(indices)
    train_idx, val_idx = indices[:n_train], indices[n_train:]
    print(f"  Train split: {n_train}  |  Val split: {n_val}")

    # ── 5. build model ────────────────────────────────────────────────────────
    print(f"\n── Step 5: Build GraphRouterModel  "
          f"(hidden={hidden_dim}, layers={num_layers}, heads={heads}) ──")
    # query in_dim = Longformer output dim (768 by default)
    # model in_dim  = profile embedding dim (may differ from query)
    q_in_dim = h_q_all.size(1)
    m_in_dim = h_m.size(1)
    if align_model and q_in_dim != m_in_dim:
        print(f"  align_model=True: mapping model features {m_in_dim}→{q_in_dim} "
              f"before shared projection")
    elif align_model and q_in_dim == m_in_dim:
        print(f"  align_model=True but dims already match ({q_in_dim}), "
              f"m_align will be identity (skipped)")

    model = GraphRouterModel(
        q_in_dim=q_in_dim, m_in_dim=m_in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers, heads=heads, dropout=dropout,
        align_model=align_model,
    ).to(device)
    total_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_p:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    # ── 6. training loop ──────────────────────────────────────────────────────
    print(f"\n── Step 6: Training  (epochs={epochs}, batch={batch_size}) ──")
    best_val_perf = -1.0
    best_state    = None

    for epoch in range(1, epochs + 1):

        # ── train ─────────────────────────────────────────────────────────────
        model.train()
        train_loss, n_b = 0.0, 0
        random.shuffle(train_idx)

        for start in range(0, len(train_idx), batch_size):
            batch = train_idx[start : start + batch_size]
            if not batch:
                continue

            # fresh encode each batch
            h_q, h_m_enc = model.encode(h_q_all, h_m, edge_index)

            batch_local_idx = torch.tensor(
                [disc_meta[gi]["local_q_idx"] for gi in batch], device=device
            )
            logits = model.edge_pred(h_q[batch_local_idx], h_m_enc)  # [B, N_m]

            labels = torch.zeros_like(logits)
            for lb, gi in enumerate(batch):
                pd = disc_meta[gi]["perf_per_model"]
                bs = max(pd.values())
                for mn, p in pd.items():
                    if p == bs and mn in model_name2idx:
                        labels[lb, model_name2idx[mn]] = 1.0

            loss = F.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_b        += 1

        avg_train = train_loss / max(n_b, 1)

        # ── validate ──────────────────────────────────────────────────────────
        model.eval()
        val_loss, n_correct, n_val_total = 0.0, 0, 0
        val_perfs: list[float] = []

        with torch.no_grad():
            h_q_v, h_m_v = model.encode(h_q_all, h_m, edge_index)

            for start in range(0, len(val_idx), batch_size):
                batch = val_idx[start : start + batch_size]
                if not batch:
                    continue

                batch_local_idx = torch.tensor(
                    [disc_meta[gi]["local_q_idx"] for gi in batch], device=device
                )
                logits = model.edge_pred(h_q_v[batch_local_idx], h_m_v)

                labels = torch.zeros_like(logits)
                for lb, gi in enumerate(batch):
                    pd = disc_meta[gi]["perf_per_model"]
                    bs = max(pd.values())
                    for mn, p in pd.items():
                        if p == bs and mn in model_name2idx:
                            labels[lb, model_name2idx[mn]] = 1.0

                val_loss  += F.binary_cross_entropy_with_logits(logits, labels).item()
                predicted  = logits.argmax(1)
                n_correct  += sum(
                    labels[i, predicted[i]].item() == 1.0
                    for i in range(len(predicted))
                )
                n_val_total += len(batch)

                for lb, gi in enumerate(batch):
                    sel   = model_names[predicted[lb].item()]
                    perf  = disc_meta[gi]["perf_per_model"].get(sel)
                    if perf is not None:
                        val_perfs.append(perf)

        avg_val     = val_loss / max(len(val_idx) // batch_size + 1, 1)
        val_acc     = n_correct / max(n_val_total, 1)
        val_avg_prf = sum(val_perfs) / len(val_perfs) if val_perfs else 0.0
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            # gradient diagnostics
            grad_norms = {
                n: p.grad.abs().mean().item()
                for n, p in model.named_parameters()
                if p.grad is not None
            }
            mean_grad = sum(grad_norms.values()) / len(grad_norms) if grad_norms else 0.0
            no_grad   = [n for n, p in model.named_parameters()
                         if p.requires_grad and p.grad is None]
            if no_grad:
                print(f"  ⚠️  No grad: {no_grad}")
            print(f"  Epoch {epoch:4d}/{epochs}  "
                  f"train={avg_train:.4f}  val_loss={avg_val:.4f}  "
                  f"val_acc={val_acc:.4f}  val_perf={val_avg_prf:.4f}  "
                  f"mean_grad={mean_grad:.2e}  lr={optimizer.param_groups[0]['lr']:.2e}")

        if val_avg_prf > best_val_perf:
            best_val_perf = val_avg_prf
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            os.makedirs(os.path.dirname(save_ckpt_path) or ".", exist_ok=True)
            torch.save({
                "epoch": epoch, "model_state": best_state,
                "val_perf": val_avg_prf, "val_acc": val_acc,
                "config": {
                    "in_dim": in_dim, "hidden_dim": hidden_dim,
                    "num_layers": num_layers, "heads": heads,
                    "model_names": model_names,
                },
            }, save_ckpt_path)
            if epoch % 10 == 0 or epoch == 1:
                print(f"    ✅ Best checkpoint saved (val_perf={val_avg_prf:.4f})")

    print(f"\n  Best val_perf: {best_val_perf:.4f}")

    # ── 7. inference on test queries ──────────────────────────────────────────
    print(f"\n── Step 7: Test inference ────────────────────────────────")
    model.load_state_dict(best_state)
    model.eval()

    selected_names: list[str] = []
    with torch.no_grad():
        h_q_t, h_m_t = model.encode(h_q_all, h_m, edge_index)
        for meta in test_meta:
            q_local = torch.tensor([meta["local_q_idx"]], device=device)
            scores  = model.edge_pred(h_q_t[q_local], h_m_t).squeeze(0)
            selected_names.append(model_names[scores.argmax().item()])

    # ── 8. evaluate + save ────────────────────────────────────────────────────
    print(f"\n── Step 8: Evaluation ────────────────────────────────────")
    results = _evaluate(test_meta, selected_names)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to '{output_path}'")
    print(f"✅ Done!  Best val_perf={best_val_perf:.4f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import os
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    _PR = os.path.join(ROOT_DIR, "routeprofile")
    _RD = os.path.join(ROOT_DIR, "route_data")

    parser = argparse.ArgumentParser(
        description="GraphRouter: bipartite GAT + edge prediction for LLM routing."
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
                             "(default: routeprofile/routing_result/{mode}/GraphRouter_results.json)")
    parser.add_argument("--save-ckpt",   default=None,
                        help="Checkpoint path "
                             "(default: routeprofile/routing_evaluation/trained_GraphRouter/{mode}/graphrouter_ckpt.pt)")
    parser.add_argument("--cache-train", default=None,
                        help="Cache path for train query embeddings (.npy)")
    parser.add_argument("--cache-test",  default=None,
                        help="Cache path for test query embeddings (.npy)")
    parser.add_argument("--hidden-dim",  type=int,   default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--num-layers",  type=int,   default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--heads",       type=int,   default=DEFAULT_HEADS)
    parser.add_argument("--dropout",     type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--lr",          type=float, default=DEFAULT_LR)
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size",  type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed",        type=int,   default=DEFAULT_SEED)
    parser.add_argument("--val-split",   type=float, default=0.1)
    parser.add_argument("--align-model-feat", action="store_true", default=False,
                        help="Add a linear layer to map model profile features into "
                             "query feature space before the shared GNN projection. "
                             "Useful when model profiles come from a different encoder "
                             "than Longformer (e.g. hetero propagation outputs).")
    args = parser.parse_args()

    _result_dir = os.path.join(_PR, "routing_result", args.mode)
    _ckpt_dir   = os.path.join(_PR, "routing_evaluation", "trained_GraphRouter", args.mode)
    os.makedirs(_result_dir, exist_ok=True)
    os.makedirs(_ckpt_dir,   exist_ok=True)

    train(
        profiles_path=args.profiles    or os.path.join(_PR, "model_profile_result", args.mode, "flat.npz"),
        train_data_path=args.train_data or os.path.join(_RD, "routing_train_data.json"),
        test_data_path=args.test_data   or os.path.join(_RD, "routing_test_data.json"),
        output_path=args.output         or os.path.join(_result_dir, "GraphRouter_results.json"),
        save_ckpt_path=args.save_ckpt   or os.path.join(_ckpt_dir, "graphrouter_ckpt.pt"),
        cache_train=args.cache_train,
        cache_test=args.cache_test,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        heads=args.heads,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        val_split=args.val_split,
        align_model=args.align_model_feat,
    )


if __name__ == "__main__":
    main()