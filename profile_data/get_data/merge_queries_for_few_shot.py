import ast
import json
import os
from typing import Any


# Optional dependency from your project.
# If unavailable, the script falls back to raw query text.
try:
    from llmrouter.utils import generate_task_query  # type: ignore
except Exception:
    generate_task_query = None


INPUT_EXISTING = 'all_dataset_queries_old.json'
INPUT_TRAIN = 'default_query_train.jsonl'
OUTPUT_PATH = 'all_dataset_queries_updated.json'


def _format_query(original_query: str, row_task_name: str | None, row: dict) -> dict[str, str | None]:
    """Reuse the same prompt formatting logic as inference, with safe fallback."""
    if row_task_name and generate_task_query is not None:
        try:
            raw_choices = row.get('choices', None)
            if isinstance(raw_choices, str):
                choices_list = ast.literal_eval(raw_choices)
            elif isinstance(raw_choices, list):
                choices_list = raw_choices
            else:
                choices_list = None
            sample_data = {'query': original_query, 'choices': choices_list}
            return generate_task_query(row_task_name, sample_data)
        except Exception:
            pass
    return {'system': None, 'user': original_query}


def load_json(path: str) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main(existing_path: str = INPUT_EXISTING, train_path: str = INPUT_TRAIN, output_path: str = OUTPUT_PATH) -> None:
    existing = load_json(existing_path)
    if not isinstance(existing, dict):
        raise ValueError('all_dataset_queries.json must be a JSON object like {"task": ["query1", ...]}')

    existing_tasks = set(existing.keys())
    rows = load_jsonl(train_path)

    result: dict[str, list[str]] = {k: v[:] for k, v in existing.items()}
    added_tasks: list[str] = []

    # Only add tasks that do NOT already exist in all_dataset_queries.json
    grouped_new: dict[str, list[str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue

        task_name = row.get('task_name')
        if not task_name or task_name in existing_tasks:
            continue

        original_query = row.get('query', '')
        if not isinstance(original_query, str) or not original_query.strip():
            continue

        formatted = _format_query(original_query, task_name, row)
        user_prompt = formatted.get('user')
        if not isinstance(user_prompt, str) or not user_prompt.strip():
            continue

        grouped_new.setdefault(task_name, []).append(user_prompt)

    # Deduplicate within each new task while preserving order
    for task_name, prompts in grouped_new.items():
        seen = set()
        deduped = []
        for p in prompts:
            if p not in seen:
                seen.add(p)
                deduped.append(p)
        result[task_name] = deduped
        added_tasks.append(task_name)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f'Loaded existing tasks: {len(existing_tasks)}')
    print(f'Loaded train rows: {len(rows)}')
    print(f'Added new tasks: {len(added_tasks)}')
    if added_tasks:
        print('New task names:')
        for name in added_tasks:
            print(f'  - {name} ({len(result[name])} queries)')
    print(f'Saved to: {output_path}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Append missing dataset queries from train jsonl into all_dataset_queries.json')
    parser.add_argument('--existing', default=INPUT_EXISTING, help='Path to all_dataset_queries.json')
    parser.add_argument('--train', default=INPUT_TRAIN, help='Path to default_query_train.jsonl')
    parser.add_argument('--output', default=OUTPUT_PATH, help='Output JSON path')
    args = parser.parse_args()

    main(args.existing, args.train, args.output)
