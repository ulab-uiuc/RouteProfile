#!/bin/bash
# =============================================================================
# Step 3b: Trainable GNN Model Profiles
# =============================================================================
# Trains a HANConv encoder via self-supervised masked feature reconstruction
# (GraphMAE-style) on the heterogeneous data graph, then extracts model
# profile embeddings from the trained encoder.
#
# Pretraining objective
# ─────────────────────
# Two complementary self-supervised tasks run jointly:
#
#   Node reconstruction  – each node's Longformer embedding (.x) is randomly
#     masked with a learned per-type [MASK] token. The encoder must reconstruct
#     the original embedding for every masked position (MSE loss).
#
#   Edge score prediction – each model↔dataset edge carries a normalised
#     benchmark score. A fraction of these scores are zeroed out; a small MLP
#     decoder must predict the hidden scores from the encoder's src/dst
#     embeddings (MSE loss).
#
# Total loss = node_reconstruction_loss + edge_loss_weight × edge_score_loss
#
# The encoder sees the corrupted graph and must learn representations that
# support both tasks. After training, the best checkpoint (lowest total loss)
# is used to extract final model-node embeddings as the profile.
#
# Usage:
#   bash routeprofile/scripts/step3b_trainable_profile.sh [MODE] [OPTIONS...]
#
# Arguments:
#   MODE     : standard (default) | newllm
#   OPTIONS  : any extra flags forwarded to trainable_gnn_profile.py
#
# Examples:
#   bash routeprofile/scripts/step3b_trainable_profile.sh standard
#   bash routeprofile/scripts/step3b_trainable_profile.sh newllm --epochs 200
#   bash routeprofile/scripts/step3b_trainable_profile.sh standard \
#       --graph profile_data/result_data_graph/standard/task_domain_graph_full.pt \
#       --hidden-dim 512 --heads 8 --num-layers 3 --epochs 200
#
# Outputs:
#   routeprofile/model_profile_result/{mode}/trainable_gnn.npz
#       Model profile embeddings (dim=768, compatible with all routers)
#   routeprofile/get_model_profile/trainable/trained_gnn/{mode}/pretrain_ckpt.pt
#       Full HANConv checkpoint (encoder + decoders + mask tokens)
#
# =============================================================================
# Parameter Reference
# =============================================================================
#
# ── Input graph ───────────────────────────────────────────────────────────────
#
#   --graph PATH         HeteroData .pt graph file to train on.
#                        default: profile_data/result_data_graph/{mode}/task_graph_full.pt
#
#                        The graph choice determines both:
#                          (a) which edge types the encoder attends over
#                          (b) which edges carry benchmark scores for the
#                              edge reconstruction loss
#
#                        task_graph_full.pt           (arch, model, dataset)
#                          Smallest graph: 2 edge types (model↔arch,
#                          model↔dataset). Edge loss trains on model↔dataset
#                          benchmark scores only.
#
#                        task_domain_graph_full.pt    (arch, model, dataset, domain)
#                          Adds domain nodes and domain↔dataset edges.
#                          At 2+ hops the encoder absorbs domain-level signals,
#                          which may produce more semantically clustered profiles.
#
#                        query_task_domain_graph_full.pt  (all 5 node types)
#                          Richest graph; query nodes link each model to concrete
#                          example inputs. Highest signal but also highest memory
#                          and compute cost.
#
# ── HANConv architecture ──────────────────────────────────────────────────────
#
#   --hidden-dim INT     Width of intermediate HANConv layers.
#                        default: 256
#                        Wider → more expressive encoder → higher GPU memory.
#                        Reduce (e.g. --hidden-dim 128) if OOM on small GPU.
#                        Increase (e.g. --hidden-dim 512) for richer profiles
#                        on larger graphs (task_domain or query_task_domain).
#
#   --out-dim INT        Dimensionality of the final model embeddings.
#                        default: 768  (must equal Longformer's output dim)
#                        Changing this requires re-encoding text nodes with a
#                        different encoder, so leave at 768 unless you swap
#                        the text encoder used in graph construction.
#
#   --heads INT          Number of attention heads in intermediate HANConv layers.
#                        default: 4
#                        More heads → finer-grained meta-path attention (each head
#                        can specialise on a different meta-path). Diminishing
#                        returns beyond 8; must divide hidden-dim evenly.
#                        The final layer always uses 1 head (outputs out-dim).
#
#   --num-layers INT     Depth of the HANConv encoder (number of message-passing
#                        rounds).
#                        default: 2
#                        Layer 1 : model embeddings absorb direct neighbours
#                                  (architectures and benchmark datasets).
#                        Layer 2 : 2-hop context; e.g. datasets pull in other
#                                  models they are connected to, giving indirect
#                                  cross-model comparison signal.
#                        3+      : deeper context but risk of over-smoothing;
#                                  useful mainly on the richer graph variants.
#
# ── Self-supervised masking ───────────────────────────────────────────────────
#
#   --node-mask-rate FLOAT
#                        Fraction of nodes whose Longformer embedding is replaced
#                        by a learned [MASK] token before encoding.
#                        default: 0.30  (30 % of all nodes, across all types)
#                        Higher rate → harder pretraining task → potentially
#                        stronger representations, but noisier gradients.
#                        Lower (e.g. 0.15) → easier task, faster convergence.
#                        Typical range: 0.15 – 0.50.
#
#   --edge-mask-rate FLOAT
#                        Fraction of model↔dataset benchmark score edges whose
#                        score is zeroed out before encoding.
#                        default: 0.30
#                        These are the only edges with scalar attributes; higher
#                        masking forces the encoder to infer scores from node
#                        context, encouraging performance-aware representations.
#                        Set to 0.0 to disable edge reconstruction entirely.
#
#   --edge-loss-w FLOAT  Weight applied to the edge reconstruction MSE when
#                        computing the total loss.
#                        default: 1.0
#                        total_loss = node_loss + edge_loss_w × edge_loss
#                        Increase (e.g. 2.0 – 5.0) to emphasise benchmark-score
#                        fidelity → profiles that cluster models by performance.
#                        Decrease (e.g. 0.1) to de-emphasise scores and rely
#                        more on text-structure signals from node reconstruction.
#                        Set to 0.0 to train with node reconstruction only
#                        (equivalent to ignoring benchmark scores).
#
# ── Training hyperparameters ──────────────────────────────────────────────────
#
#   --lr FLOAT           Initial learning rate for AdamW.
#                        default: 1e-4
#                        A cosine annealing schedule decays lr to 5 % of its
#                        initial value by the final epoch.
#                        Larger lr (e.g. 3e-4) converges faster but may overshoot;
#                        smaller (e.g. 5e-5) is safer on smaller graphs.
#
#   --epochs INT         Number of full training passes over the graph.
#                        default: 100
#                        Loss is typically stable by epoch 80–120 for the default
#                        task_graph_full.pt. Richer graphs (domain, query variants)
#                        may benefit from 150–200 epochs.
#                        Loss and lr are printed every 10 epochs; the best
#                        checkpoint (lowest total loss) is saved automatically.
#
#   --seed INT           Random seed for PyTorch and Python (reproducibility).
#                        default: 42
#                        Controls mask sampling, weight initialisation, and
#                        AdamW initialisation. Change to get a different
#                        run for ensemble averaging.
#
# ── Output filtering ──────────────────────────────────────────────────────────
#
#   --keep [M ...]       Space-separated list of model names to save in the
#                        output .npz file.
#                        Omit flag → uses the default 8-model candidate set.
#                        --keep with no args → save every model in the graph.
#                        Example: --keep qwen2.5-7b-instruct llama-3.1-8b-instruct
#
#   --save-emb PATH      Override output path for the model profile .npz.
#                        default: routeprofile/model_profile_result/{mode}/trainable_gnn.npz
#
#   --save-ckpt PATH     Override output path for the full training checkpoint.
#                        default: routeprofile/get_model_profile/trainable/
#                                 trained_gnn/{mode}/pretrain_ckpt.pt
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAINABLE_DIR="${PROJECT_ROOT}/routeprofile/get_model_profile/trainable"

MODE="${1:-standard}"
shift || true   # remaining args forwarded to the python script

if [ "${MODE}" != "standard" ] && [ "${MODE}" != "newllm" ]; then
    echo "Error: MODE must be 'standard' or 'newllm'. Got: '${MODE}'"
    exit 1
fi

echo ""
echo "============================================================"
echo " Step 3b: Trainable GNN Profiles  |  mode=${MODE}"
echo "============================================================"
echo "Training HANConv encoder with masked feature reconstruction ..."
echo ""

python "${TRAINABLE_DIR}/trainable_gnn_profile.py" \
    --mode "${MODE}" \
    "$@"

echo ""
echo "✅ Profile saved to : ${PROJECT_ROOT}/routeprofile/model_profile_result/${MODE}/trainable_gnn.npz"
echo "✅ Checkpoint saved to: ${PROJECT_ROOT}/routeprofile/get_model_profile/trainable/trained_gnn/${MODE}/pretrain_ckpt.pt"
