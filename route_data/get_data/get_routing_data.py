"""
all_model_inference.py
----------------------
Calls every model for every query and saves results incrementally.

Features
--------
- Checkpoint every N queries (default 100) via atomic overwrite
- Resume support: if the output file already exists, completed queries
  are loaded and skipped automatically
- Gemma system-prompt fix included
"""

import ast
import argparse
import copy
import json
import os
from typing import Any

from tqdm import tqdm

from llmrouter.utils import call_api, generate_task_query, calculate_task_performance


# ── helpers ────────────────────────────────────────────────────────────────────

def _save_atomic(data: list[dict], path: str) -> None:
    """Overwrite path atomically via a .tmp file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _format_query(
    original_query: str,
    row_task_name:  str | None,
    row:            dict,
) -> dict[str, str | None]:
    """Format the query for execution, falling back to raw on failure."""
    if row_task_name:
        try:
            raw_choices = row.get("choices", None)
            if isinstance(raw_choices, str):
                choices_list = ast.literal_eval(raw_choices)
            elif isinstance(raw_choices, list):
                choices_list = raw_choices
            else:
                choices_list = None
            sample_data = {"query": original_query, "choices": choices_list}
            return generate_task_query(row_task_name, sample_data)
        except (ValueError, KeyError) as e:
            print(f"  Warning: prompt formatting failed ({e}). Using raw query.")
    return {"system": None, "user": original_query}


def _apply_model_fixes(
    model_name: str,
    query_text: dict[str, str | None],
) -> dict[str, str | None]:
    """
    Apply any per-model prompt fixes before calling the API.
    Returns a (possibly modified) copy of query_text.
    """
    qt = copy.copy(query_text)

    # Gemma does not support system prompts — merge into user turn
    if model_name == "gemma-2-9b-it":
        if qt.get("system") is not None:
            qt["user"]   = qt["system"] + "\n\n" + (qt.get("user") or "")
            qt["system"] = None

    return qt


def _call_one_model(
    model_name:    str,
    model_info:    dict,
    query_text:    dict[str, str | None],
    row_task_name: str | None,
    ground_truth:  str | None,
    metric:        str | None,
) -> dict[str, Any]:
    """Call one model API and return a populated model_performance entry."""
    info           = model_info[model_name]
    api_model_name = info.get("model", model_name)
    api_endpoint   = info.get("api_endpoint")
    service        = info.get("service")

    fixed_query = _apply_model_fixes(model_name, query_text)

    request: dict[str, Any] = {
        "api_endpoint": api_endpoint,
        "query":        fixed_query,
        "model_name":   model_name,
        "api_name":     api_model_name,
    }
    if service:
        request["service"] = service

    try:
        result            = call_api(request, max_tokens=1024, temperature=0.0)
        response          = result.get("response", "")
        prompt_tokens     = result.get("prompt_tokens", 0)
        completion_tokens = result.get("completion_tokens", 0)
        success           = "error" not in result
    except Exception as e:
        print(f"    ✗ API call failed for '{model_name}': {e}")
        response, prompt_tokens, completion_tokens, success = "", 0, 0, False

    entry: dict[str, Any] = {
        "response":          response,
        "prompt_tokens":     prompt_tokens,
        "completion_tokens": completion_tokens,
        "input_token":       prompt_tokens,
        "output_token":      completion_tokens,
        "success":           success,
    }

    if ground_truth:
        perf = calculate_task_performance(
            prediction=response,
            ground_truth=ground_truth,
            task_name=row_task_name,
            metric=metric,
        )
        if perf is not None:
            entry["task_performance"] = perf

    return entry

# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    default_query_path = os.path.join(
        project_root, "data", "default_query_train.jsonl"
    )
    default_model_path = os.path.join(
        project_root, "data", "model_info_cleaned.json"
    )
    default_output_path = os.path.join(
        project_root, "routing_data", "routing_train_data.json",
    )

    parser = argparse.ArgumentParser(
        description="All-model inference with checkpointing and resume support."
    )
    parser.add_argument(
        "--query_data_test_path",
        default=default_query_path,
        help="Path to the query .jsonl file",
    )
    parser.add_argument(
        "--model_path",
        default=default_model_path,
        help="Path to model_info JSON",
    )
    parser.add_argument(
        "--output_path",
        default=default_output_path,
        help="Path to save (and resume from) the output JSON",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Save a checkpoint every N completed queries (default: 100)",
    )
    args = parser.parse_args()

    # ── load query data ────────────────────────────────────────────────────────
    print(f"Loading queries from: '{args.query_data_test_path}' ...")
    query_data: list[dict] = []
    with open(args.query_data_test_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                query_data.append(json.loads(line))
    print(f"  {len(query_data)} queries loaded.")

    # ── load model info ────────────────────────────────────────────────────────
    print(f"Loading model info from: '{args.model_path}' ...")
    with open(args.model_path, "r", encoding="utf-8") as f:
        model_info: dict = json.load(f)
    print(f"  {len(model_info)} models: {list(model_info.keys())}")

    # ── resume: load existing output if present ────────────────────────────────
    completed_output: list[dict] = []
    n_completed = 0

    if os.path.exists(args.output_path):
        print(f"\nExisting output found at '{args.output_path}'. Resuming ...")
        with open(args.output_path, "r", encoding="utf-8") as f:
            completed_output = json.load(f)
        n_completed = len(completed_output)
        print(f"  {n_completed} queries already completed — skipping.")
    else:
        print("\nNo existing output found. Starting from scratch.")

    remaining_queries = query_data[n_completed:]
    if not remaining_queries:
        print("All queries already processed. Nothing to do.")
        return

    print(f"  {len(remaining_queries)} queries remaining.\n")

    # ── inference loop ─────────────────────────────────────────────────────────
    output: list[dict] = completed_output   # grow this list in-place

    since_last_checkpoint = 0

    for row in tqdm(remaining_queries, desc="Processing queries",
                    initial=n_completed, total=len(query_data)):

        if isinstance(row, dict):
            row_copy       = copy.copy(row)
            original_query = row_copy.get("query", "")
            row_task_name  = row_copy.get("task_name")
        else:
            row_copy       = {"query": str(row)}
            original_query = str(row)
            row_task_name  = None

        ground_truth = (
            row_copy.get("ground_truth")
            or row_copy.get("gt")
            or row_copy.get("answer")
        )
        metric = row_copy.get("metric")

        # format query once, shared across all models for this row
        query_text = _format_query(original_query, row_task_name, row_copy)
        row_copy["formatted_query"] = query_text

        # call every model
        row_copy["model_performance"] = {}
        for model_name in model_info:
            entry = _call_one_model(
                model_name, model_info, query_text,
                row_task_name, ground_truth, metric,
            )
            row_copy["model_performance"][model_name] = entry

        output.append(row_copy)
        since_last_checkpoint += 1

        # ── checkpoint ────────────────────────────────────────────────────────
        if since_last_checkpoint >= args.checkpoint_every:
            _save_atomic(output, args.output_path)
            tqdm.write(f"  💾 Checkpoint saved ({len(output)} queries done).")
            since_last_checkpoint = 0

    # ── final save ─────────────────────────────────────────────────────────────
    _save_atomic(output, args.output_path)
    print(f"\n💾 Final results saved to: '{args.output_path}'")
    print(f"   Total queries processed: {len(output)}")


if __name__ == "__main__":
    main()