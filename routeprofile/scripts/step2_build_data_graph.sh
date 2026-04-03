#!/bin/bash
# =============================================================================
# Step 2: Build Data Graphs
# =============================================================================
# Constructs all heterogeneous PyG graphs from profile_data/ JSON files.
# Outputs are saved to results/result_data_graph/{mode}/.
#
# Usage:
#   bash routeprofile/scripts/step2_build_data_graph.sh [MODE]
#
# Arguments:
#   MODE  : standard (default) | newllm | both
#           - standard : uses model_feature_standard.json + task_queries_standard.json
#           - newllm   : uses model_feature_newllm.json   + task_queries_newllm.json
#           - both     : runs both standard and newllm in sequence
#
# Examples:
#   bash routeprofile/scripts/step2_build_data_graph.sh standard
#   bash routeprofile/scripts/step2_build_data_graph.sh newllm
#   bash routeprofile/scripts/step2_build_data_graph.sh both
#
# Output graphs (per mode):
#   results/result_data_graph/{mode}/task_graph_full.pt
#   results/result_data_graph/{mode}/query_graph_full.pt
#   results/result_data_graph/{mode}/query_task_graph_full.pt
#   results/result_data_graph/{mode}/task_domain_graph_full.pt
#   results/result_data_graph/{mode}/query_task_domain_graph_full.pt
# =============================================================================

set -e

# Locate project root (two levels above this script: scripts/ -> routeprofile/ -> RouteProfile/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/routeprofile/build_data_graph"

MODE="${1:-standard}"

build_for_mode() {
    local mode="$1"
    echo ""
    echo "============================================================"
    echo " Building graphs for mode: ${mode}"
    echo "============================================================"

    echo "[1/5] Building task graph ..."
    python "${BUILD_DIR}/build_task_graph.py" --mode "${mode}"

    echo "[2/5] Building query graph ..."
    python "${BUILD_DIR}/build_query_graph.py" --mode "${mode}"

    echo "[3/5] Building query-task graph ..."
    python "${BUILD_DIR}/build_query_task_graph.py" --mode "${mode}"

    echo "[4/5] Building task-domain graph ..."
    python "${BUILD_DIR}/build_task_domain_graph.py" --mode "${mode}"

    echo "[5/5] Building query-task-domain graph ..."
    python "${BUILD_DIR}/build_query_task_domain_graph.py" --mode "${mode}"

    echo ""
    echo "✅ All graphs for mode '${mode}' saved to:"
    echo "   ${PROJECT_ROOT}/results/result_data_graph/${mode}/"
}

if [ "${MODE}" = "both" ]; then
    build_for_mode "standard"
    build_for_mode "newllm"
elif [ "${MODE}" = "standard" ] || [ "${MODE}" = "newllm" ]; then
    build_for_mode "${MODE}"
else
    echo "Error: MODE must be 'standard', 'newllm', or 'both'. Got: '${MODE}'"
    exit 1
fi
