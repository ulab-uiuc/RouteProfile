"""
EmbeddingSimRouter (offline mode)
----------------------------------
Instead of calling the API at routing time, this version reads pre-computed
responses for all models from a routing_test_data.json file and simply
looks up the response of the model selected by cosine similarity.

Input file format  (routing_test_data.json)
-------------------------------------------
[
  {
    "query":        "...",
    "task_name":    "...",
    "ground_truth": "...",
    "metric":       "...",
    "choices":      "...",
    "model_performance": {
      "model-name-a": {
        "response":         "...",
        "task_performance": float | missing,
        "success":          bool,
        ...
      },
      ...
    }
  },
  ...
]

Output file format  (results/*.json)
--------------------------------------
{
  "performance_summary": { "task_name": avg_performance, ... },
  "routing_results": [
    {
      ...original fields...,
      "model_name":       "selected-model-name",
      "similarity_scores": { "model": score, ... },
      "response":         "...",
      "task_performance": float | None,
      "success":          bool,
    },
    ...
  ]
}
"""

import json
import copy
import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from llmrouter.models.meta_router import MetaRouter
import ast

from llmrouter.utils import get_longformer_embedding, calculate_task_performance, generate_task_query
import argparse


def _format_query_text(
    original_query: str,
    task_name:      str | None,
    row:            dict,
) -> str:
    """
    Format the query using generate_task_query (same logic as all_model_inference).

    Returns the user-turn string that should be embedded for routing.
    Falls back to the raw query string if formatting fails.
    """
    if task_name:
        try:
            raw_choices = row.get("choices", None)
            if isinstance(raw_choices, str):
                choices_list = ast.literal_eval(raw_choices)
            elif isinstance(raw_choices, list):
                choices_list = raw_choices
            else:
                choices_list = None
            sample_data   = {"query": original_query, "choices": choices_list}
            formatted     = generate_task_query(task_name, sample_data)
            # use the user turn for embedding; fall back to original if empty
            user_text = formatted.get("user") or ""
            system_text = formatted.get("system") or ""
            # combine system + user so the embedding sees the full context
            combined = (system_text + "\n\n" + user_text).strip() if system_text else user_text.strip()
            return combined if combined else original_query
        except (ValueError, KeyError, Exception) as e:
            print(f"  Warning: query formatting failed ({e}). Using raw query.")
    return original_query


class EmbeddingSimRouter(MetaRouter):
    """
    Offline training-free router.

    Selects a model per query by cosine similarity between the query embedding
    and pre-computed model embeddings, then looks up the pre-computed response
    from routing_test_data.json instead of calling any API.
    """

    def __init__(self, yaml_path: str, model_embeddings_path: str):
        dummy_model = nn.Identity()
        super().__init__(model=dummy_model, yaml_path=yaml_path)

        # ── load model embeddings ──────────────────────────────────────────────
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        emb_path = os.path.join(
            project_root,
            model_embeddings_path,
        )
        self.model_names, self.model_emb_matrix = self._load_embeddings(emb_path)

        print(
            f"  Loaded {len(self.model_names)} model embeddings "
            f"(dim={self.model_emb_matrix.shape[1]}) from '{emb_path}'"
        )
        print(f"  Models: {self.model_names}")

        # ── validate llm_data covers all embedding keys ────────────────────────
        if hasattr(self, "llm_data") and self.llm_data:
            missing = [n for n in self.model_names if n not in self.llm_data]
            if missing:
                raise KeyError(
                    f"Model names from .npz not found in llm_data: {missing}. "
                    "Check llm_data_path in the YAML."
                )
        else:
            print("Warning: llm_data not loaded.")

    # ── embedding loading ──────────────────────────────────────────────────────

    @staticmethod
    def _load_embeddings(npz_path: str) -> tuple[list[str], torch.Tensor]:
        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Model embeddings not found: '{npz_path}'")

        loaded      = np.load(npz_path)
        model_names = list(loaded.files)
        emb_matrix  = np.stack([loaded[n] for n in model_names])

        emb_tensor = torch.from_numpy(emb_matrix).float()
        emb_tensor = F.normalize(emb_tensor, p=2, dim=1)

        return model_names, emb_tensor

    # ── core similarity routing ────────────────────────────────────────────────

    def _select_model(self, query_embedding: torch.Tensor) -> tuple[str, torch.Tensor]:
        """Return (best_model_name, all_scores) via cosine similarity."""
        query_norm = F.normalize(query_embedding.float().unsqueeze(0), p=2, dim=1)
        scores     = (query_norm @ self.model_emb_matrix.T).squeeze(0)
        best_idx   = scores.argmax().item()
        return self.model_names[best_idx], scores

    # ── MetaRouter interface ───────────────────────────────────────────────────

    def route_single(self, query: dict[str, Any]) -> dict[str, Any]:
        """
        Route a single query and return it enriched with routing metadata.
        Actual response lookup requires the pre-computed data; use route_batch
        when you have the full routing_test_data available.
        """
        formatted_query        = _format_query_text(
            query["query"],
            query.get("task_name"),
            query,
        )
        query_embedding        = get_longformer_embedding(formatted_query)
        model_name, scores     = self._select_model(query_embedding)

        output                 = copy.copy(query)
        output["model_name"]   = model_name
        output["similarity_scores"] = {
            name: round(scores[i].item(), 6)
            for i, name in enumerate(self.model_names)
        }
        return output

    def route_batch(
        self,
        batch:     Optional[Any] = None,
        task_name: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Route every query in `batch` (which must be the pre-computed
        routing_test_data list) and look up the selected model's response.

        Args:
            batch     : list of dicts loaded from routing_test_data.json,
                        each containing a "model_performance" key.
                        Pass None to fall back to self.query_data_test
                        (but that won't have model_performance).
            task_name : unused; kept for interface compatibility.

        Returns:
            List of output dicts with "model_name", "similarity_scores",
            "response", "task_performance", and "success" filled in from
            the pre-computed data.
        """
        # ── determine input data ───────────────────────────────────────────────
        if batch is not None:
            query_data = batch if isinstance(batch, list) else [batch]
        else:
            if hasattr(self, "query_data_test") and self.query_data_test is not None:
                query_data = copy.copy(self.query_data_test)
            else:
                print("Warning: No batch provided and no test data available.")
                return []

        query_data_output: list[dict[str, Any]] = []

        for row in query_data:
            if isinstance(row, dict):
                row_copy       = copy.copy(row)
                original_query = row_copy.get("query", "")
                row_task_name  = row_copy.get("task_name", task_name)
            else:
                row_copy       = {"query": str(row)}
                original_query = str(row)
                row_task_name  = task_name

            # ── step 1: select model via cosine similarity ─────────────────────
            # Format the query the same way all_model_inference does before
            # embedding — this ensures routing uses the same text the models see.
            formatted_query    = _format_query_text(original_query, row_task_name, row_copy)
            query_embedding    = get_longformer_embedding(formatted_query)
            model_name, scores = self._select_model(query_embedding)

            row_copy["model_name"]        = model_name
            row_copy["similarity_scores"] = {
                name: round(scores[i].item(), 6)
                for i, name in enumerate(self.model_names)
            }

            # ── step 2: look up pre-computed response ──────────────────────────
            model_performance = row_copy.get("model_performance", {})

            if model_name not in model_performance:
                print(
                    f"Warning: model '{model_name}' not found in model_performance "
                    f"for query '{original_query[:60]}...'. Skipping."
                )
                row_copy.update({
                    "response":         "",
                    "task_performance": None,
                    "success":          False,
                })
                query_data_output.append(row_copy)
                continue

            selected = model_performance[model_name]

            row_copy["response"]          = selected.get("response", "")
            row_copy["prompt_tokens"]     = selected.get("prompt_tokens", 0)
            row_copy["completion_tokens"] = selected.get("completion_tokens", 0)
            row_copy["input_token"]       = selected.get("input_token", 0)
            row_copy["output_token"]      = selected.get("output_token", 0)
            row_copy["success"]           = selected.get("success", False)

            # ── step 3: use pre-computed task_performance if available,
            #            otherwise recompute from ground truth ─────────────────
            if "task_performance" in selected and selected["task_performance"] is not None:
                row_copy["task_performance"] = selected["task_performance"]
            else:
                ground_truth = (
                    row_copy.get("ground_truth")
                    or row_copy.get("gt")
                    or row_copy.get("answer")
                )
                metric = row_copy.get("metric")
                if ground_truth and row_copy["success"]:
                    task_performance = calculate_task_performance(
                        prediction=row_copy["response"],
                        ground_truth=ground_truth,
                        task_name=row_task_name,
                        metric=metric,
                    )
                    row_copy["task_performance"] = task_performance
                else:
                    row_copy["task_performance"] = None

            # ── step 4: compute is_hit and is_discriminative ──────────────────
            # Gather all task_performance values for this query across all models.
            all_perfs: list[float] = [
                v.get("task_performance")
                for v in model_performance.values()
                if v.get("task_performance") is not None
            ]

            # is_discriminative: True iff at least one model scored 0 AND
            # at least one scored 1 (i.e. the benchmark distinguishes models)
            has_zero = any(p == 0 for p in all_perfs)
            has_one  = any(p == 1 for p in all_perfs)
            row_copy["is_discriminative"] = bool(has_zero and has_one)

            # is_hit: True if the selected model's performance is non-zero,
            # OR if every model scored 0 (nothing to distinguish, default True)
            selected_perf = row_copy.get("task_performance")
            all_zero      = all_perfs and all(p == 0 for p in all_perfs)
            if all_zero or not all_perfs:
                row_copy["is_hit"] = True
            else:
                row_copy["is_hit"] = bool(selected_perf is not None and selected_perf > 0)

            query_data_output.append(row_copy)

        return query_data_output


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    _PR = os.path.join(ROOT_DIR, "routeprofile")
    _RD = os.path.join(ROOT_DIR, "route_data")

    parser = argparse.ArgumentParser(
        description="Offline EmbeddingSimRouter: route queries using pre-computed responses."
    )
    parser.add_argument("--mode", choices=["standard", "newllm"], default="standard",
                        help="Routing setting: standard or newllm (default: standard)")
    parser.add_argument(
        "--yaml_path",
        type=str,
        default=None,
        help="Path to the YAML config file (optional)",
    )
    parser.add_argument(
        "--routing_data_path",
        type=str,
        default=None,
        help="Path to routing_test_data.json (default: route_data/routing_test_data.json)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save the routing output JSON "
             "(default: routeprofile/routing_result/{mode}/SimRouter_results.json)",
    )
    parser.add_argument(
        "--model_embeddings_path",
        type=str,
        default=None,
        help="Path to the .npz file containing model embeddings "
             "(default: routeprofile/model_profile_result/{mode}/flat.npz)",
    )
    args = parser.parse_args()

    # ── resolve default paths ──────────────────────────────────────────────────
    routing_data_path    = args.routing_data_path    or os.path.join(_RD, "routing_test_data.json")
    model_embeddings_path = args.model_embeddings_path or os.path.join(_PR, "model_profile_result", args.mode, "flat.npz")
    _result_dir          = os.path.join(_PR, "routing_result", args.mode)
    os.makedirs(_result_dir, exist_ok=True)
    output_path          = args.output_path or os.path.join(_result_dir, "SimRouter_results.json")

    # ── validate required paths ────────────────────────────────────────────────
    if not os.path.exists(routing_data_path):
        raise FileNotFoundError(f"routing_test_data not found: '{routing_data_path}'")
    if not os.path.exists(model_embeddings_path):
        raise FileNotFoundError(f"model embeddings not found: '{model_embeddings_path}'")

    # ── load pre-computed routing data ─────────────────────────────────────────
    print(f"Loading routing data from: '{routing_data_path}' ...")
    with open(routing_data_path, "r", encoding="utf-8") as f:
        routing_data: list[dict] = json.load(f)
    print(f"  {len(routing_data)} queries loaded.")

    # ── initialise router ──────────────────────────────────────────────────────
    print(f"\nInitialising EmbeddingSimRouter ...")
    router = EmbeddingSimRouter(args.yaml_path, model_embeddings_path)
    print("✅ Router initialised.")

    # ── run offline routing ────────────────────────────────────────────────────
    print("\nRouting queries (no API calls) ...")
    routing_result = router.route_batch(batch=routing_data)

    # ── compute performance summary ────────────────────────────────────────────
    performance_summary: dict[str, list[float]] = {}
    hit_summary:         dict[str, list[bool]]  = {}
    for item in routing_result:
        t    = item.get("task_name", "unknown")
        perf = item.get("task_performance")
        hit  = item.get("is_hit")
        if perf is not None:
            performance_summary.setdefault(t, []).append(perf)
        if hit is not None:
            hit_summary.setdefault(t, []).append(hit)

    task_performance_summary: dict[str, float] = {}
    hit_performance_summary:  dict[str, float] = {}
    for task, perfs in performance_summary.items():
        avg     = sum(perfs) / len(perfs)
        hits    = hit_summary.get(task, [])
        avg_hit = sum(hits) / len(hits) if hits else None
        task_performance_summary[task] = avg
        if avg_hit is not None:
            hit_performance_summary[task] = avg_hit
        hit_str = f"  avg_hit={avg_hit:.4f}" if avg_hit is not None else ""
        print(f"  📊 {task}: {avg:.4f}  (n={len(perfs)}){hit_str}")

    # ── top-level aggregate metrics (not inside performance_summary) ───────────
    all_perfs_flat  = [p for perfs in performance_summary.values() for p in perfs]
    avg_performance = sum(all_perfs_flat) / len(all_perfs_flat) if all_perfs_flat else None

    all_hits_flat = [h for hits in hit_summary.values() for h in hits]
    total_avg_hit = sum(all_hits_flat) / len(all_hits_flat) if all_hits_flat else None

    if avg_performance is not None:
        print(f"  📊 avg_performance : {avg_performance:.4f}  (n={len(all_perfs_flat)})")
    if total_avg_hit is not None:
        print(f"  📊 total_avg_hit   : {total_avg_hit:.4f}  (n={len(all_hits_flat)})")

    # ── save results ───────────────────────────────────────────────────────────
    result = {
        "avg_performance": avg_performance,
        "total_avg_hit":   total_avg_hit,
        "performance_summary": {
            "task_performance": task_performance_summary,
            "hit_performance":  hit_performance_summary,
        },
        "routing_results": routing_result,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Results saved to: '{output_path}'")


if __name__ == "__main__":
    main()