#!/bin/bash
# =============================================================================
# Step 3a: Training-Free Model Profiles
# =============================================================================
# Generates model profile embeddings using training-free methods.
# Reads heterogeneous graphs from profile_data/result_data_graph/{mode}/ and
# writes .npz embedding files to routeprofile/model_profile_result/{mode}/.
#
# Four methods are available, each producing a different kind of profile:
#
#   flat     – Aggregates the model's own text with randomly sampled neighbour
#              texts from the graph, then encodes the concatenation with
#              Longformer. Simple but captures neighbourhood context as text.
#
#   index    – Assigns each model a random vector. A lower-bound baseline.
#              Useful for sanity-checking the routing pipeline.
#
#   emb_gnn  – Training-free neighbourhood propagation on the graph.
#              Applies K rounds of degree-normalised message passing (AX)
#              so each model embedding absorbs information from its k-hop
#              neighbourhood. The model↔dataset edges are weighted by the
#              normalised benchmark scores, giving stronger signal from
#              datasets where the model performs distinctively.
#
#   text_gnn – LLM-based text aggregation. Each hop calls a local vLLM
#              model to write a new natural-language summary for each model
#              node, fusing its current text with neighbour texts. The final
#              summary is encoded by Longformer. Rich but GPU-intensive.
#              Requires vLLM installed and a suitable GPU.
#
# Usage:
#   bash routeprofile/scripts/step3a_training_free_profile.sh [MODE] [METHOD] [OPTIONS...]
#
# Arguments:
#   MODE    : standard (default) | newllm
#   METHOD  : all (default) | flat | index | emb_gnn | text_gnn
#   OPTIONS : additional flags forwarded directly to the Python script
#             (see per-method parameter reference below)
#
# Examples:
#   bash routeprofile/scripts/step3a_training_free_profile.sh standard all
#   bash routeprofile/scripts/step3a_training_free_profile.sh newllm emb_gnn
#   bash routeprofile/scripts/step3a_training_free_profile.sh standard emb_gnn --K 3 --norm right
#   bash routeprofile/scripts/step3a_training_free_profile.sh standard flat --top-k 10 --seed 0
#   bash routeprofile/scripts/step3a_training_free_profile.sh standard text_gnn --K 2 --tp 2
#
# Output (.npz files per mode):
#   routeprofile/model_profile_result/{mode}/flat.npz
#   routeprofile/model_profile_result/{mode}/index.npz
#   routeprofile/model_profile_result/{mode}/emb_gnn.npz
#   routeprofile/model_profile_result/{mode}/text_gnn.npz  (text_gnn only)
#
# =============================================================================
# Per-Method Parameter Reference
# =============================================================================
#
# ── flat (flat_profile.py) ────────────────────────────────────────────────────
#
#   --graph PATH         Input HeteroData graph .pt file.
#                        default: profile_data/result_data_graph/{mode}/task_graph_full.pt
#                        Tip: use a graph with more node types (e.g. task_domain_graph_full.pt
#                        or query_task_domain_graph_full.pt) to include richer neighbour
#                        context (domain descriptions, example queries) in the concatenated text.
#
#   --top-k INT          Number of nodes to randomly sample from each neighbour node type.
#                        default: 5
#                        Higher k → richer concatenated context → longer text → slower Longformer
#                        encoding, but potentially more discriminative profiles.
#
#   --seed INT           Random seed for reproducible neighbour sampling.
#                        default: 42
#
#   --batch-size INT     Number of concatenated texts encoded by Longformer per forward pass.
#                        default: 32  (reduce if OOM)
#
#   --save PATH          Override output .npz path.
#                        default: routeprofile/model_profile_result/{mode}/flat.npz
#
#   --keep [M ...]       Space-separated list of model names to include in the output.
#                        Omit flag → uses the default 8-model set.
#                        --keep with no args → include every model in the graph.
#
# ── index (index_profile.py) ──────────────────────────────────────────────────
#
#   --seed INT           Random seed for vector generation.
#                        default: 56
#                        Changing the seed produces a different random baseline.
#
#   --save PATH          Override output .npz path.
#                        default: routeprofile/model_profile_result/{mode}/index.npz
#
# ── emb_gnn (emb_gnn_profile.py) ─────────────────────────────────────────────
#
#   --graph PATH         Input HeteroData graph .pt file.
#                        default: profile_data/result_data_graph/{mode}/task_graph_full.pt
#                        Tip: graphs with more node types give richer multi-hop propagation.
#                        task_domain_graph_full.pt adds domain↔dataset edges so model
#                        embeddings also absorb domain-level signals after 2+ hops.
#
#   --K INT              Number of message-passing hops.
#                        default: 2
#                        K=1 : each model embedding aggregates its direct neighbours
#                              (architectures and datasets it has scores on).
#                        K=2 : adds 2-hop context (e.g. datasets reach back through
#                              other models, giving indirect cross-model comparisons).
#                        K=0 : returns the raw Longformer embeddings without propagation.
#                        Higher K smooths representations; too high can over-smooth.
#
#   --norm {sym|right|left|none}
#                        Adjacency normalisation for message passing.
#                        default: sym
#                        sym   : D_dst^{-1/2} · A · D_src^{-1/2}  (GCN-style; balanced)
#                        right : A · D_src^{-1}  (source-degree normalised; row-stochastic)
#                        left  : D_dst^{-1} · A  (destination-degree normalised)
#                        none  : raw adjacency, no normalisation (may magnify high-degree nodes)
#
#   --normalize          If set, L2-normalise the output embeddings.
#                        Useful when comparing profiles with cosine similarity (SimRouter).
#
#   --save PATH          Override output .npz path.
#                        default: routeprofile/model_profile_result/{mode}/emb_gnn.npz
#
#   --keep [M ...]       Model names to save (same semantics as --keep in flat).
#
# ── text_gnn (text_gnn_profile.py) ───────────────────────────────────────────
#   NOTE: requires a GPU and vLLM installed (`pip install vllm`).
#
#   --graph PATH         Input HeteroData graph .pt file.
#                        default: profile_data/result_data_graph/{mode}/query_task_domain_graph_full.pt
#                        Tip: this method benefits from the richest graph because the LLM
#                        prompt is built from neighbour node_feature_texts of all types.
#                        Using query_task_domain_graph_full.pt gives the LLM access to
#                        example queries, dataset descriptions, and domain context.
#
#   --K INT              Number of text-GNN hops (LLM calls per model per hop).
#                        default: 4
#                        K=0 : no LLM is called; the original model text is encoded
#                              directly by Longformer (text-only baseline, fast).
#                        K=1 : one round of LLM-based neighbourhood summarisation.
#                        K=4 : four rounds; profile text captures multi-hop context
#                              but requires 4× more LLM inference time.
#
#   --model MODEL_ID     vLLM model to use for text aggregation.
#                        default: Qwen/Qwen2.5-7B-Instruct
#                        Any model loadable by vLLM works; larger models may produce
#                        better summaries at higher compute cost.
#
#   --max-tokens INT     Maximum new tokens per LLM call.
#                        default: 500  (roughly 2-4 sentences)
#
#   --temperature FLOAT  LLM sampling temperature.
#                        default: 0.0  (greedy decoding; fully reproducible)
#
#   --tp INT             Tensor parallel size (number of GPUs for model parallelism).
#                        default: 1  (single GPU)
#                        Increase for large vLLM models (e.g. --tp 4 for 70B models).
#
#   --emb-save PATH      Override output path for Longformer embeddings (.npz).
#                        default: routeprofile/model_profile_result/{mode}/text_gnn.npz
#
#   --text-save PATH     Override output path for generated text summaries (.json).
#                        default: routeprofile/model_profile_result/{mode}/text_gnn_texts.json
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROFILE_DIR="${PROJECT_ROOT}/routeprofile/get_model_profile/training_free"

MODE="${1:-standard}"
METHOD="${2:-all}"
shift 2 2>/dev/null || true   # remaining args forwarded to the python script

if [ "${MODE}" != "standard" ] && [ "${MODE}" != "newllm" ]; then
    echo "Error: MODE must be 'standard' or 'newllm'. Got: '${MODE}'"
    exit 1
fi

run_flat() {
    echo "[flat] Generating flat (random-sampled text) profiles ..."
    python "${PROFILE_DIR}/flat_profile.py" --mode "${MODE}" "$@"
    echo "[flat] Done."
}

run_index() {
    echo "[index] Generating random index (baseline) profiles ..."
    python "${PROFILE_DIR}/index_profile.py" --mode "${MODE}" "$@"
    echo "[index] Done."
}

run_emb_gnn() {
    echo "[emb_gnn] Generating training-free GNN profiles (default: K=2, norm=sym) ..."
    python "${PROFILE_DIR}/emb_gnn_profile.py" --mode "${MODE}" --K 2 --norm sym "$@"
    echo "[emb_gnn] Done."
}

run_text_gnn() {
    echo "[text_gnn] Generating text-GNN profiles (requires vLLM + GPU) ..."
    echo "           Default: query_task_domain_graph_full.pt, K=4, Qwen2.5-7B-Instruct"
    python "${PROFILE_DIR}/text_gnn_profile.py" --mode "${MODE}" --K 4 "$@"
    echo "[text_gnn] Done."
}

echo ""
echo "============================================================"
echo " Step 3a: Training-Free Profiles  |  mode=${MODE}, method=${METHOD}"
echo "============================================================"

case "${METHOD}" in
    flat)     run_flat     "$@" ;;
    index)    run_index    "$@" ;;
    emb_gnn)  run_emb_gnn  "$@" ;;
    text_gnn) run_text_gnn "$@" ;;
    all)
        run_flat
        run_index
        run_emb_gnn
        echo ""
        echo "NOTE: text_gnn was skipped (requires vLLM). Run explicitly:"
        echo "      bash step3a_training_free_profile.sh ${MODE} text_gnn"
        ;;
    *)
        echo "Error: METHOD must be flat | index | emb_gnn | text_gnn | all. Got: '${METHOD}'"
        exit 1
        ;;
esac

echo ""
echo "✅ Profiles saved to: ${PROJECT_ROOT}/routeprofile/model_profile_result/${MODE}/"
