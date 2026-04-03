#!/bin/bash
# =============================================================================
# Step 4: Routing Evaluation
# =============================================================================
# Evaluates LLM routing performance using one of three routers:
#   SimRouter   – similarity-based (training-free)
#   MLPRouter   – pairwise-ranking MLP (trainable)
#   GraphRouter – bipartite GAT with edge prediction (trainable)
#
# Usage:
#   bash routeprofile/scripts/step4_routing_evaluation.sh [MODE] [ROUTER] [PROFILE] [OPTIONS...]
#
# Arguments:
#   MODE    : standard (default) | newllm
#   ROUTER  : sim (default) | mlp | graph | all
#             - sim   : SimRouter  (no training required)
#             - mlp   : MLPRouter  (trains and evaluates)
#             - graph : GraphRouter (trains and evaluates)
#             - all   : runs sim, mlp, graph in sequence
#   PROFILE : profile .npz filename inside results/model_profile_result/{mode}/
#             default: flat.npz
#             other options: index.npz | emb_gnn.npz | trainable_gnn.npz | text_gnn.npz
#   OPTIONS : extra flags forwarded to the selected router script
#
# Examples:
#   # SimRouter with flat profiles (standard mode)
#   bash routeprofile/scripts/step4_routing_evaluation.sh standard sim flat.npz
#
#   # MLPRouter with emb_gnn profiles (newllm mode)
#   bash routeprofile/scripts/step4_routing_evaluation.sh newllm mlp emb_gnn.npz
#
#   # GraphRouter with trainable GNN profiles
#   bash routeprofile/scripts/step4_routing_evaluation.sh standard graph trainable_gnn.npz
#
#   # Run all three routers with flat profiles
#   bash routeprofile/scripts/step4_routing_evaluation.sh standard all flat.npz
#
# Outputs (per mode and router):
#   results/routing_result/{mode}/SimRouter_results.json
#   results/routing_result/{mode}/MLPRouter_results.json
#   results/routing_result/{mode}/GraphRouter_results.json
#
# Trained checkpoints:
#   results/trained_MLPRouter/{mode}/mlp_router_ckpt.pt
#   results/trained_GraphRouter/{mode}/graphrouter_ckpt.pt
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
EVAL_DIR="${PROJECT_ROOT}/routeprofile/routing_evaluation"

MODE="${1:-standard}"
ROUTER="${2:-sim}"
PROFILE_FILE="${3:-flat.npz}"
shift 3 2>/dev/null || true   # remaining args forwarded to router script

if [ "${MODE}" != "standard" ] && [ "${MODE}" != "newllm" ]; then
    echo "Error: MODE must be 'standard' or 'newllm'. Got: '${MODE}'"
    exit 1
fi

PROFILE_PATH="${PROJECT_ROOT}/results/model_profile_result/${MODE}/${PROFILE_FILE}"

run_sim() {
    echo "[SimRouter] Running similarity-based routing ..."
    python "${EVAL_DIR}/SimRouter.py" \
        --mode "${MODE}" \
        --model_embeddings_path "${PROFILE_PATH}" \
        "$@"
    echo "[SimRouter] Done."
}

run_mlp() {
    echo "[MLPRouter] Training and evaluating MLP router ..."
    python "${EVAL_DIR}/MLPRouter.py" \
        --mode "${MODE}" \
        --profiles "${PROFILE_PATH}" \
        "$@"
    echo "[MLPRouter] Done."
}

run_graph() {
    echo "[GraphRouter] Training and evaluating Graph router ..."
    python "${EVAL_DIR}/GraphRouter.py" \
        --mode "${MODE}" \
        --profiles "${PROFILE_PATH}" \
        "$@"
    echo "[GraphRouter] Done."
}

echo ""
echo "============================================================"
echo " Step 4: Routing Evaluation  |  mode=${MODE}, router=${ROUTER}, profile=${PROFILE_FILE}"
echo "============================================================"

if [ ! -f "${PROFILE_PATH}" ]; then
    echo "Error: Profile file not found: '${PROFILE_PATH}'"
    echo "       Run Step 3a or 3b first to generate model profiles."
    exit 1
fi

case "${ROUTER}" in
    sim)   run_sim   "$@" ;;
    mlp)   run_mlp   "$@" ;;
    graph) run_graph "$@" ;;
    all)
        run_sim   "$@"
        run_mlp   "$@"
        run_graph "$@"
        ;;
    *)
        echo "Error: ROUTER must be sim | mlp | graph | all. Got: '${ROUTER}'"
        exit 1
        ;;
esac

echo ""
echo "✅ Routing results saved to: ${PROJECT_ROOT}/results/routing_result/${MODE}/"
