"""
Utility script to inspect a saved HeteroData graph.

Usage:
    python print_graph.py --graph profile_data/result_data_graph/standard/query_task_domain_graph_full.pt
"""

import argparse
import os
import torch
from collections import Counter

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def main(graph_path: str) -> None:
    data = torch.load(graph_path, weights_only=False)

    print(f"\nGraph loaded from: '{graph_path}'")
    print(data)

    print(f"\n--- Node counts ---")
    for ntype in data.node_types:
        print(f"  {ntype:20s}: {data[ntype].num_nodes} nodes")

    print(f"\n--- Edge counts ---")
    for etype in data.edge_types:
        print(f"  {str(etype):60s}: {data[etype].edge_index.shape[1]} edges")

    # If dataset→query edges exist, print per-dataset query counts
    if ("dataset", "dataset_to_query", "query") in data.edge_types:
        edge_index = data["dataset", "dataset_to_query", "query"].edge_index
        dataset_src = edge_index[0].tolist()
        count = Counter(dataset_src)
        dataset_names = data["dataset"].node_names
        print(f"\n--- Queries per dataset ---")
        print(f"  {'dataset':<30} {'#queries':>8}")
        print("  " + "-" * 40)
        for i, name in enumerate(dataset_names):
            n = count.get(i, 0)
            print(f"  {name:<30} {n:>8}")
        print("  " + "-" * 40)
        print(f"  {'total':<30} {sum(count.values()):>8}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Print summary of a saved HeteroData graph.")
    parser.add_argument(
        "--graph", required=True, metavar="PATH",
        help="Path to the .pt graph file (e.g. profile_data/result_data_graph/standard/query_task_domain_graph_full.pt)"
    )
    args = parser.parse_args()
    main(args.graph)