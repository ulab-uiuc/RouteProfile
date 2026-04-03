"""Verify that every public symbol in routeprofile is importable."""


def test_version():
    import routeprofile
    assert routeprofile.__version__ == "0.1.0"


def test_build_graph_functions():
    from routeprofile import (
        build_task_graph,
        build_query_graph,
        build_query_task_graph,
        build_task_domain_graph,
        build_query_task_domain_graph,
    )
    assert callable(build_task_graph)
    assert callable(build_query_graph)
    assert callable(build_query_task_graph)
    assert callable(build_task_domain_graph)
    assert callable(build_query_task_domain_graph)


def test_profile_functions():
    from routeprofile import (
        build_flat_profile,
        build_emb_gnn_profile,
        build_index_profile,
        build_text_gnn_profile,
        build_trainable_gnn_profile,
    )
    assert callable(build_flat_profile)
    assert callable(build_emb_gnn_profile)
    assert callable(build_index_profile)
    assert callable(build_text_gnn_profile)
    assert callable(build_trainable_gnn_profile)


def test_router_functions_and_classes():
    from routeprofile import (
        call_simrouter,
        SimRouter,
        call_mlprouter,
        MLPRouter,
        call_graphrouter,
        GraphRouter,
    )
    assert callable(call_simrouter)
    assert callable(call_mlprouter)
    assert callable(call_graphrouter)
    # Classes should be importable
    assert SimRouter is not None
    assert MLPRouter is not None
    assert GraphRouter is not None


def test_subpackage_imports():
    from routeprofile.build_data_graph import build_task_graph
    from routeprofile.get_model_profile.training_free import (
        build_flat_profile,
        build_emb_gnn_profile,
        build_index_profile,
        build_text_gnn_profile,
    )
    from routeprofile.get_model_profile.trainable import build_trainable_gnn_profile
    from routeprofile.routing_evaluation import (
        call_simrouter, SimRouter,
        call_mlprouter, MLPRouter,
        call_graphrouter, GraphRouter,
    )
