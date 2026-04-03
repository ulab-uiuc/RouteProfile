"""
pretrain_han_masked.py
-----------------------
Self-supervised pretraining of a HANConv encoder on a HeteroData graph
via masked feature reconstruction (inspired by GraphMAE).

Masking strategy
----------------
  node features : each node's .x is independently replaced with a learned
                  [MASK] token with probability `node_mask_rate`.
  edge features : each model<->dataset edge's score (edge_attr) is
                  independently zeroed with probability `edge_mask_rate`.

Reconstruction targets
----------------------
  masked nodes : encoder output → per-type MLP decoder → predicted .x
                 loss = MSE(predicted, original_x)   [masked nodes only]
  masked edges : concat(src_emb, dst_emb) → edge MLP decoder → predicted score
                 loss = MSE(predicted, original_score) [masked edges only]

Total loss = node_loss + edge_loss_weight * edge_loss

After pretraining, model node embeddings are extracted and saved as .npz,
compatible with EmbeddingSimRouter.

Usage
-----
    python pretrain_han_masked.py \\
        --graph      /path/to/hetero_graph.pt \\
        --save-emb   /path/to/pretrained_model_emb.npz \\
        --save-ckpt  /path/to/pretrain_ckpt.pt
"""

import argparse
import os
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv


# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_HIDDEN_DIM      = 256
DEFAULT_OUT_DIM         = 768
DEFAULT_HEADS           = 4
DEFAULT_NUM_LAYERS      = 2
DEFAULT_NODE_MASK_RATE  = 0.30
DEFAULT_EDGE_MASK_RATE  = 0.30
DEFAULT_EDGE_LOSS_W     = 1.0
DEFAULT_LR              = 1e-4
DEFAULT_EPOCHS          = 100
DEFAULT_SEED            = 42
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


# ── HANConv encoder ───────────────────────────────────────────────────────────

class HANEncoder(nn.Module):
    """
    K-layer HANConv encoder.

    All node types share the same in_channels dimension (Longformer dim).
    Output embeddings are L2-normalised.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim:  int,
        out_dim:     int,
        heads:       int,
        num_layers:  int,
        metadata:    tuple,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}.")

        layers = []
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            in_ch   = in_channels if i == 0 else hidden_dim
            out_ch  = out_dim     if is_last else hidden_dim
            n_heads = 1           if is_last else heads
            layers.append(HANConv(
                in_channels=in_ch,
                out_channels=out_ch,
                heads=n_heads,
                dropout=0.1,
                metadata=metadata,
            ))

        self.convs      = nn.ModuleList(layers)
        self.num_layers = num_layers

    def forward(
        self,
        x_dict:          dict[str, Tensor],
        edge_index_dict: dict,
    ) -> dict[str, Tensor]:
        h = x_dict
        for i, conv in enumerate(self.convs):
            is_last = (i == self.num_layers - 1)
            h_new   = conv(h, edge_index_dict)
            h = {}
            for ntype in (h_new or {}):
                if h_new[ntype] is not None:
                    h[ntype] = h_new[ntype] if is_last else F.elu(h_new[ntype])
                else:
                    # HANConv returned None → fall back to previous representation
                    if ntype in (h_new or {}):
                        pass   # already handled above
            # carry forward any node type that HANConv skipped entirely
            for ntype in x_dict:
                if ntype not in h:
                    h[ntype] = list(x_dict.values())[0].new_zeros(
                        x_dict[ntype].size(0),
                        list(h.values())[0].size(1) if h else x_dict[ntype].size(1)
                    )

        return {k: F.normalize(v, p=2, dim=1) for k, v in h.items()}


# ── Per-type MLP decoders ─────────────────────────────────────────────────────

class NodeDecoder(nn.Module):
    """
    Per-node-type linear decoder: out_dim → in_dim (reconstruct original .x).
    """
    def __init__(self, in_dim: int, out_dim: int, node_types: list[str]) -> None:
        super().__init__()
        self.decoders = nn.ModuleDict({
            ntype: nn.Sequential(
                nn.Linear(out_dim, out_dim),
                nn.ELU(),
                nn.Linear(out_dim, in_dim),
            )
            for ntype in node_types
        })

    def forward(self, ntype: str, emb: Tensor) -> Tensor:
        return self.decoders[ntype](emb)


class EdgeDecoder(nn.Module):
    """
    Edge score decoder: concat(src_emb, dst_emb) → scalar score in [0, 1].
    Used only for model<->dataset edges that carry edge_attr.
    """
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.ELU(),
            nn.Linear(out_dim, 1),
            nn.Sigmoid(),   # scores are already normalised to [0,1] during masking
        )

    def forward(self, src_emb: Tensor, dst_emb: Tensor) -> Tensor:
        return self.mlp(torch.cat([src_emb, dst_emb], dim=1)).squeeze(1)


# ── Masking helpers ────────────────────────────────────────────────────────────

def _mask_node_features(
    x_dict:    dict[str, Tensor],
    mask_rate: float,
    mask_tokens: dict[str, Tensor],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """
    Randomly replace node features with per-type learned mask tokens.

    Returns:
        masked_x_dict  : x_dict with masked positions replaced by mask token
        node_mask_dict : { ntype: BoolTensor [N] } True = masked
    """
    masked_x  = {}
    node_masks = {}
    for ntype, x in x_dict.items():
        mask = torch.rand(x.size(0), device=x.device) < mask_rate   # [N]
        x_masked = x.clone()
        if mask.any():
            x_masked[mask] = mask_tokens[ntype].to(x.device)
        masked_x[ntype]   = x_masked
        node_masks[ntype] = mask
    return masked_x, node_masks


def _mask_edge_features(
    edge_index_dict: dict,
    edge_attr_dict:  dict,
    mask_rate:       float,
) -> tuple[dict, dict, dict]:
    """
    Randomly zero out edge_attr values for model<->dataset edges.

    Returns:
        masked_attr_dict : edge_attr with masked values set to 0
        edge_mask_dict   : { etype_key: BoolTensor [E] } True = masked
        orig_attr_dict   : original (unmasked) edge_attr values
    """
    masked_attr = {}
    edge_masks  = {}
    orig_attrs  = {}

    for etype, attr in edge_attr_dict.items():
        mask = torch.rand(attr.size(0), device=attr.device) < mask_rate
        a_masked = attr.clone()
        a_masked[mask] = 0.0
        masked_attr[etype] = a_masked
        edge_masks[etype]  = mask
        orig_attrs[etype]  = attr

    return masked_attr, edge_masks, orig_attrs


# ── Training ───────────────────────────────────────────────────────────────────

def train(
    graph_path:      str,
    save_emb_path:   str,
    save_ckpt_path:  str,
    hidden_dim:      int,
    out_dim:         int,
    heads:           int,
    num_layers:      int,
    node_mask_rate:  float,
    edge_mask_rate:  float,
    edge_loss_w:     float,
    lr:              float,
    epochs:          int,
    seed:            int,
    keep_names:      list[str],
) -> None:

    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── 1. load graph ──────────────────────────────────────────────────────────
    print(f"\n── Step 1: Load graph from '{graph_path}' ────────────────")
    data: HeteroData = torch.load(graph_path, weights_only=False)
    print(data)

    node_types = list(data.node_types)
    in_dim     = data[node_types[0]].x.float().size(1)
    print(f"  Node types  : {node_types}")
    print(f"  in_dim      : {in_dim}")

    if in_dim != out_dim:
        raise ValueError(
            f"out_dim ({out_dim}) must equal Longformer dim ({in_dim}) "
            "so that model embeddings are compatible with EmbeddingSimRouter."
        )

    # move everything to device
    x_dict: dict[str, Tensor] = {
        ntype: data[ntype].x.float().to(device)
        for ntype in node_types
    }
    edge_index_dict: dict = {
        etype: data[etype].edge_index.to(device)
        for etype in data.edge_types
        if hasattr(data[etype], "edge_index")
    }

    # collect edge types that carry edge_attr (model<->dataset score edges)
    scored_etypes: list = [
        etype for etype in data.edge_types
        if hasattr(data[etype], "edge_attr") and data[etype].edge_attr is not None
    ]
    edge_attr_dict: dict[tuple, Tensor] = {
        etype: data[etype].edge_attr.float().to(device)
        for etype in scored_etypes
    }

    # min-max normalise edge scores to [0,1] so MSE is on a consistent scale
    for etype in scored_etypes:
        attr = edge_attr_dict[etype].squeeze(1)
        a_min, a_max = attr.min(), attr.max()
        if (a_max - a_min).abs() > 1e-8:
            edge_attr_dict[etype] = ((attr - a_min) / (a_max - a_min)).unsqueeze(1)
        else:
            edge_attr_dict[etype] = torch.ones_like(attr).unsqueeze(1)

    # ── 2. build models ────────────────────────────────────────────────────────
    print(f"\n── Step 2: Build HANEncoder + decoders ───────────────────")

    encoder = HANEncoder(
        in_channels=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        heads=heads,
        num_layers=num_layers,
        metadata=data.metadata(),
    ).to(device)

    # per-type learnable [MASK] tokens
    mask_tokens = nn.ParameterDict({
        ntype: nn.Parameter(torch.randn(in_dim, device=device) * 0.02)
        for ntype in node_types
    })

    node_decoder = NodeDecoder(
        in_dim=in_dim,
        out_dim=out_dim,
        node_types=node_types,
    ).to(device)

    edge_decoder = EdgeDecoder(out_dim=out_dim).to(device)

    all_params = (
        list(encoder.parameters())
        + list(mask_tokens.values())
        + list(node_decoder.parameters())
        + list(edge_decoder.parameters())
    )
    total_params = sum(p.numel() for p in all_params)
    print(f"  Encoder params      : {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"  Mask token params   : {sum(p.numel() for p in mask_tokens.values()):,}")
    print(f"  Decoder params      : {sum(p.numel() for p in node_decoder.parameters()) + sum(p.numel() for p in edge_decoder.parameters()):,}")
    print(f"  Total trainable     : {total_params:,}")

    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.05
    )

    # ── 3. pretraining loop ───────────────────────────────────────────────────
    print(f"\n── Step 3: Pretraining  "
          f"(epochs={epochs}, node_mask={node_mask_rate}, "
          f"edge_mask={edge_mask_rate}, edge_loss_w={edge_loss_w}) ──")

    best_loss    = float("inf")
    best_state   = None

    for epoch in range(1, epochs + 1):

        encoder.train()
        node_decoder.train()
        edge_decoder.train()
        for p in mask_tokens.values():
            p.requires_grad_(True)

        # ── node masking ───────────────────────────────────────────────────────
        masked_x, node_masks = _mask_node_features(
            x_dict, node_mask_rate, mask_tokens
        )

        # ── edge masking ───────────────────────────────────────────────────────
        masked_edge_attr, edge_masks, orig_edge_attr = _mask_edge_features(
            edge_index_dict, edge_attr_dict, edge_mask_rate
        )

        # build edge_index_dict with updated edge_attr baked into the graph
        # (HANConv does not use edge_attr, so we only need the masked x_dict)
        emb_dict = encoder(masked_x, edge_index_dict)

        # ── node reconstruction loss ───────────────────────────────────────────
        node_loss = torch.tensor(0.0, device=device)
        n_node_types_with_mask = 0

        for ntype in node_types:
            mask = node_masks[ntype]
            if not mask.any():
                continue
            emb_masked = emb_dict.get(ntype)
            if emb_masked is None:
                continue
            pred = node_decoder(ntype, emb_masked[mask])   # [N_masked, in_dim]
            tgt  = x_dict[ntype][mask]                     # [N_masked, in_dim]
            node_loss = node_loss + F.mse_loss(pred, tgt)
            n_node_types_with_mask += 1

        if n_node_types_with_mask > 0:
            node_loss = node_loss / n_node_types_with_mask

        # ── edge reconstruction loss ───────────────────────────────────────────
        edge_loss = torch.tensor(0.0, device=device)
        n_scored_etypes_with_mask = 0

        for etype in scored_etypes:
            mask = edge_masks[etype]             # [E] bool
            if not mask.any():
                continue

            src_type, rel, dst_type = etype
            ei = edge_index_dict[etype]          # [2, E]

            src_emb = emb_dict.get(src_type)
            dst_emb = emb_dict.get(dst_type)
            if src_emb is None or dst_emb is None:
                continue

            src_masked_idx = ei[0][mask]
            dst_masked_idx = ei[1][mask]

            pred_scores = edge_decoder(
                src_emb[src_masked_idx],
                dst_emb[dst_masked_idx],
            )                                    # [E_masked]
            tgt_scores  = orig_edge_attr[etype].squeeze(1)[mask]   # [E_masked]

            edge_loss = edge_loss + F.mse_loss(pred_scores, tgt_scores)
            n_scored_etypes_with_mask += 1

        if n_scored_etypes_with_mask > 0:
            edge_loss = edge_loss / n_scored_etypes_with_mask

        total_loss = node_loss + edge_loss_w * edge_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{epochs}  "
                  f"total={total_loss.item():.4f}  "
                  f"node={node_loss.item():.4f}  "
                  f"edge={edge_loss.item():.4f}  "
                  f"lr={current_lr:.2e}")

        if total_loss.item() < best_loss:
            best_loss  = total_loss.item()
            best_state = {
                "encoder":      {k: v.cpu().clone() for k, v in encoder.state_dict().items()},
                "node_decoder": {k: v.cpu().clone() for k, v in node_decoder.state_dict().items()},
                "edge_decoder": {k: v.cpu().clone() for k, v in edge_decoder.state_dict().items()},
                "mask_tokens":  {k: v.detach().cpu().clone() for k, v in mask_tokens.items()},
            }

    print(f"\n  Best total loss: {best_loss:.4f}")

    # ── 4. save checkpoint ────────────────────────────────────────────────────
    print(f"\n── Step 4: Save checkpoint → '{save_ckpt_path}' ─────────")
    checkpoint = {
        "best_loss":    best_loss,
        "encoder":      best_state["encoder"],
        "node_decoder": best_state["node_decoder"],
        "edge_decoder": best_state["edge_decoder"],
        "mask_tokens":  best_state["mask_tokens"],
        "config": {
            "in_dim":          in_dim,
            "hidden_dim":      hidden_dim,
            "out_dim":         out_dim,
            "heads":           heads,
            "num_layers":      num_layers,
            "node_mask_rate":  node_mask_rate,
            "edge_mask_rate":  edge_mask_rate,
            "metadata":        data.metadata(),
        },
    }
    os.makedirs(os.path.dirname(save_ckpt_path) or ".", exist_ok=True)
    torch.save(checkpoint, save_ckpt_path)
    print(f"  Checkpoint saved.")

    # ── 5. extract model embeddings ───────────────────────────────────────────
    print(f"\n── Step 5: Extract model embeddings → '{save_emb_path}' ─")
    encoder.load_state_dict(best_state["encoder"])
    encoder.eval()
    encoder.to(device)

    with torch.no_grad():
        emb_dict = encoder(x_dict, edge_index_dict)   # use ORIGINAL x (no mask)

    model_emb_tensor = emb_dict["model"].cpu()   # [N_models, out_dim]
    model_names: list[str] = list(data["model"].node_names)
    name2idx = {n: i for i, n in enumerate(model_names)}

    if keep_names:
        missing  = [n for n in keep_names if n not in name2idx]
        if missing:
            print(f"  Warning: models not found in graph: {missing}")
        selected = [(n, name2idx[n]) for n in keep_names if n in name2idx]
    else:
        selected = [(n, i) for i, n in enumerate(model_names)]

    emb_dict_np = {
        name: model_emb_tensor[idx].numpy().astype(np.float32)
        for name, idx in selected
    }
    os.makedirs(os.path.dirname(save_emb_path) or ".", exist_ok=True)
    np.savez(save_emb_path, **emb_dict_np)

    print(f"  Saved {len(emb_dict_np)} model embeddings  "
          f"(dim={model_emb_tensor.size(1)}) → '{save_emb_path}'")
    print(f"  Models: {list(emb_dict_np.keys())}")
    print(f"\n✅ Pretraining complete!")


# ── Python API ────────────────────────────────────────────────────────────────

def build_model_profile(
    mode:           str        = "standard",
    graph:          str | None = None,
    save_emb:       str | None = None,
    save_ckpt:      str | None = None,
    hidden_dim:     int        = DEFAULT_HIDDEN_DIM,
    out_dim:        int        = DEFAULT_OUT_DIM,
    heads:          int        = DEFAULT_HEADS,
    num_layers:     int        = DEFAULT_NUM_LAYERS,
    node_mask_rate: float      = DEFAULT_NODE_MASK_RATE,
    edge_mask_rate: float      = DEFAULT_EDGE_MASK_RATE,
    edge_loss_w:    float      = DEFAULT_EDGE_LOSS_W,
    lr:             float      = DEFAULT_LR,
    epochs:         int        = DEFAULT_EPOCHS,
    seed:           int        = DEFAULT_SEED,
    keep:           list | None = None,
) -> None:
    """
    Train a HANConv encoder via self-supervised masked feature reconstruction
    and save the resulting model profile embeddings.

    Args:
        mode           : "standard" or "newllm" — selects default graph/save paths
        graph          : path to HeteroData .pt graph (None → auto from mode)
        save_emb       : output .npz path for embeddings (None → auto from mode)
        save_ckpt      : output .pt path for checkpoint  (None → auto from mode)
        hidden_dim     : width of intermediate HANConv layers (default 256)
        out_dim        : output embedding dim (default 768, must match Longformer)
        heads          : attention heads in intermediate layers (default 4)
        num_layers     : number of HANConv message-passing rounds (default 2)
        node_mask_rate : fraction of node features to mask (default 0.30)
        edge_mask_rate : fraction of edge scores to mask (default 0.30)
        edge_loss_w    : weight for edge reconstruction loss (default 1.0)
        lr             : initial AdamW learning rate (default 1e-4)
        epochs         : training epochs (default 100)
        seed           : random seed (default 42)
        keep           : model names to save; None → DEFAULT_KEEP_MODELS; [] → all
    """
    import os
    ROOT_DIR  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    _PD       = os.path.join(ROOT_DIR, "profile_data")
    _RESULTS  = os.path.join(ROOT_DIR, "results")
    _emb_dir  = os.path.join(_RESULTS, "model_profile_result", mode)
    _ckpt_dir = os.path.join(_RESULTS, "trained_trainable_gnn", mode)
    os.makedirs(_emb_dir,  exist_ok=True)
    os.makedirs(_ckpt_dir, exist_ok=True)

    graph_path     = graph    or os.path.join(_RESULTS, "result_data_graph", mode, "task_graph_full.pt")
    save_emb_path  = save_emb  or os.path.join(_emb_dir,  "trainable_gnn.npz")
    save_ckpt_path = save_ckpt or os.path.join(_ckpt_dir, "pretrain_ckpt.pt")

    if keep is None:
        keep_names = DEFAULT_KEEP_MODELS
    elif len(keep) == 0:
        keep_names = []
    else:
        keep_names = keep

    train(
        graph_path=graph_path,
        save_emb_path=save_emb_path,
        save_ckpt_path=save_ckpt_path,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        heads=heads,
        num_layers=num_layers,
        node_mask_rate=node_mask_rate,
        edge_mask_rate=edge_mask_rate,
        edge_loss_w=edge_loss_w,
        lr=lr,
        epochs=epochs,
        seed=seed,
        keep_names=keep_names,
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def cli() -> None:
    import os

    parser = argparse.ArgumentParser(
        description="Self-supervised HANConv pretraining via masked feature reconstruction."
    )
    parser.add_argument("--mode",           choices=["standard", "newllm"], default="standard",
                        help="Routing setting: standard or newllm (default: standard)")
    parser.add_argument("--graph",          default=None,
                        help="Path to HeteroData .pt graph "
                             "(default: profile_data/result_data_graph/{mode}/task_graph_full.pt)")
    parser.add_argument("--save-emb",       default=None,
                        help="Output path for model embeddings "
                             "(default: routeprofile/model_profile_result/{mode}/trainable_gnn.npz)")
    parser.add_argument("--save-ckpt",      default=None,
                        help="Output path for full checkpoint "
                             "(default: routeprofile/get_model_profile/trainable/trained_gnn/{mode}/pretrain_ckpt.pt)")
    parser.add_argument("--hidden-dim",     type=int,   default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--out-dim",        type=int,   default=DEFAULT_OUT_DIM)
    parser.add_argument("--heads",          type=int,   default=DEFAULT_HEADS)
    parser.add_argument("--num-layers",     type=int,   default=DEFAULT_NUM_LAYERS)
    parser.add_argument("--node-mask-rate", type=float, default=DEFAULT_NODE_MASK_RATE,
                        help="Fraction of node features to mask (default 0.30)")
    parser.add_argument("--edge-mask-rate", type=float, default=DEFAULT_EDGE_MASK_RATE,
                        help="Fraction of edge scores to mask (default 0.30)")
    parser.add_argument("--edge-loss-w",    type=float, default=DEFAULT_EDGE_LOSS_W,
                        help="Weight of edge reconstruction loss (default 1.0)")
    parser.add_argument("--lr",             type=float, default=DEFAULT_LR)
    parser.add_argument("--epochs",         type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--seed",           type=int,   default=DEFAULT_SEED)
    parser.add_argument("--keep",           nargs="*",  default=None, metavar="MODEL",
                        help="Model names to save. Omit = DEFAULT_KEEP_MODELS; "
                             "--keep with no args = all.")
    args = parser.parse_args()

    build_model_profile(
        mode=args.mode,
        graph=args.graph,
        save_emb=args.save_emb,
        save_ckpt=args.save_ckpt,
        hidden_dim=args.hidden_dim,
        out_dim=args.out_dim,
        heads=args.heads,
        num_layers=args.num_layers,
        node_mask_rate=args.node_mask_rate,
        edge_mask_rate=args.edge_mask_rate,
        edge_loss_w=args.edge_loss_w,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        keep=args.keep,
    )


if __name__ == "__main__":
    cli()