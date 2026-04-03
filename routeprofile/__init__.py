"""RouteProfile: Heterogeneous graph-based model profiles for LLM routing."""
__version__ = "0.1.0"

from routeprofile.build_data_graph import (
    build_task_graph,
    build_query_graph,
    build_query_task_graph,
    build_task_domain_graph,
    build_query_task_domain_graph,
)
from routeprofile.get_model_profile.training_free import (
    build_flat_profile,
    build_emb_gnn_profile,
    build_index_profile,
    build_text_gnn_profile,
)
from routeprofile.get_model_profile.trainable import build_trainable_gnn_profile
from routeprofile.routing_evaluation import (
    call_simrouter,
    SimRouter,
    call_mlprouter,
    MLPRouter,
    call_graphrouter,
    GraphRouter,
)

__all__ = [
    # graph builders
    "build_task_graph",
    "build_query_graph",
    "build_query_task_graph",
    "build_task_domain_graph",
    "build_query_task_domain_graph",
    # profile builders
    "build_flat_profile",
    "build_emb_gnn_profile",
    "build_index_profile",
    "build_text_gnn_profile",
    "build_trainable_gnn_profile",
    # routers
    "call_simrouter",   "SimRouter",
    "call_mlprouter",   "MLPRouter",
    "call_graphrouter", "GraphRouter",
]
