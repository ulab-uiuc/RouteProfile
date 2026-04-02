import json
from datasets import load_dataset, get_dataset_config_names
import random

# 你关心的数据集
DATASETS = {
    "ifeval": "google/IFEval",
    "bbh": "lukaemon/bbh",
    "math": "HuggingFaceH4/MATH-500",
    "gpqa": "Idavidrein/gpqa",
    "musr": "TAUR-Lab/MuSR",
    "mmlu_pro": "TIGER-Lab/MMLU-Pro",
    "EvalPlus": "evalplus/humanevalplus",
    "MultiPL-E": "nuprl/MultiPL-E",
    "C-Eval": "ceval/ceval-exam",
    "AGIEval English": "lighteval/agi_eval_en",
    "SQuAD": "rajpurkar/squad",
    "TheoremQA": "TIGER-Lab/TheoremQA",
    "WinoGrande": "allenai/winogrande",
    "BoolQ": "google/boolq",
    "DROP": "ucinlp/drop",
    "TruthfulQA": "domenicrosati/TruthfulQA",
    "WildBench": "allenai/WildBench",
}

# 常见的 query 字段名候选
QUERY_FIELDS = [
    "query",
    "question",
    "prompt",
    "input",
    "problem",
    "instruction",
]

# 每个数据集如果字段名特殊，可以在这里手动指定
FIELD_OVERRIDE = {
    "ifeval": "prompt",
    "bbh": "input",
    "math": "problem",
    "gpqa": "Question",
    "musr": "question",
    "mmlu_pro": "question",
    
    "MultiPL-E": "prompt",
    "EvalPlus": "prompt",
    "C-Eval": "question",
    "AGIEval English": "question",
    "SQuAD": "question",
    "TheoremQA": "Question",
    "WinoGrande": "sentence",
    "BoolQ": "question",
    "DROP": "question",
    "TruthfulQA": "Question",
    "WildBench": "conversation_input",
}


def find_query_field(example: dict, dataset_name: str):
    """
    自动寻找该样本中的 query 字段
    """
    if dataset_name in FIELD_OVERRIDE:
        field = FIELD_OVERRIDE[dataset_name]
        if field in example:
            return field

    # 优先精确匹配
    for field in QUERY_FIELDS:
        if field in example:
            return field

    # 再做不区分大小写匹配
    lowered = {k.lower(): k for k in example.keys()}
    for field in QUERY_FIELDS:
        if field.lower() in lowered:
            return lowered[field.lower()]

    return None


def load_all_splits(dataset_path: str):
    """
    尝试把一个数据集的所有 split 都读出来。
    如果数据集有多个 config，也会把每个 config 的 split 都读出来。
    返回: list[dict]
    """
    all_examples = []

    try:
        # 先尝试获取 config 名
        configs = get_dataset_config_names(dataset_path)
    except Exception:
        configs = []

    if not configs:
        # 没有 config 的情况
        ds = load_dataset(dataset_path)
        for split_name in ds.keys():
            all_examples.extend(ds[split_name])
        return all_examples

    if dataset_path == "Idavidrein/gpqa":
        configs = ["gpqa_diamond"]
    elif dataset_path == "allenai/winogrande":
        configs = ["winogrande_xs"]
    elif dataset_path == "allenai/WildBench":
        configs = ["v2-hard"]

    # 有 config 的情况
    for config in configs:
        try:
            ds = load_dataset(dataset_path, config)
            if dataset_path == "TIGER-Lab/MMLU-Pro":
                all_examples.extend(ds["test"])
            elif dataset_path == "ucinlp/drop":
                all_examples.extend(ds["validation"])
            elif dataset_path == "ceval/ceval-exam":
                all_examples.extend(ds["test"])
            elif dataset_path == "lighteval/agi_eval_en":
                all_examples.extend(ds["validation"])
            elif dataset_path == "rajpurkar/squad":
                all_examples.extend(ds["validation"])
            elif dataset_path == "allenai/winogrande":
                all_examples.extend(ds["test"])
            elif dataset_path == "google/boolq":
                all_examples.extend(ds["validation"])
            else:
                for split_name in ds.keys():
                    all_examples.extend(ds[split_name])
        except Exception as e:
            print(f"[WARN] 跳过 {dataset_path} config={config}: {e}")

    return all_examples

def extract_queries(dataset_name: str, dataset_path: str, max_queries: int = 1000, seed: int = 42):
    """
    提取一个数据集中的所有 query
    - 先打乱
    - 如果数量超过 max_queries，则只保留前 max_queries 条
    """
    print(f"Processing {dataset_name} ...")
    examples = load_all_splits(dataset_path)

    if not examples:
        print(f"[WARN] {dataset_name} 没有读取到样本")
        return []

    field = find_query_field(examples[0], dataset_name)
    if field is None:
        print(f"[WARN] {dataset_name} 找不到 query 字段，可用字段: {list(examples[0].keys())}")
        return []

    queries = []
    for ex in examples:
        if dataset_name == "musr":
            queries.append(str(ex.get("narrative")) + "\n\n" + str(ex.get("question")))
        elif dataset_name == "C-Eval":
            queries.append(
                str(ex.get("question")) + "\n"
                + "A. " + str(ex.get("A")) + "\n"
                + "B. " + str(ex.get("B")) + "\n"
                + "C. " + str(ex.get("C")) + "\n"
                + "D. " + str(ex.get("D"))
            )
        elif dataset_name == "SQuAD":
            queries.append(str(ex.get("context")) + "\n\n" + str(ex.get("question")))
        elif dataset_name == "WinoGrande":
            queries.append(
                str(ex.get("sentence")) + "\n\n"
                + "Option 1: " + str(ex.get("option1")) + "\n"
                + "Option 2: " + str(ex.get("option2"))
            )
        elif dataset_name == "BoolQ":
            queries.append(str(ex.get("passage")) + "\n\n" + str(ex.get("question")))
        elif dataset_name == "DROP":
            queries.append(str(ex.get("passage")) + "\n\n" + str(ex.get("question")))
        elif dataset_name == "WildBench":
            queries.append(
                str(ex.get("conversation_input")) + "\n\n"
                + str(ex.get("references")) + "\n\n"
                + str(ex.get("checklist"))
            )
        else:
            value = ex.get(field)
            if isinstance(value, str):
                queries.append(value)
            elif value is not None:
                queries.append(str(value))

    # 打乱
    random.seed(seed)
    random.shuffle(queries)

    # 超过 max_queries 时截断
    if len(queries) > max_queries:
        queries = queries[:max_queries]

    return queries

def main():
    output = {}

    for dataset_name, dataset_path in DATASETS.items():
        try:
            output[dataset_name] = extract_queries(dataset_name, dataset_path)
            print(f"[OK] {dataset_name}: {len(output[dataset_name])} queries")
        except Exception as e:
            print(f"[ERROR] {dataset_name}: {e}")
            output[dataset_name] = []

    with open("all_dataset_queries_old.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("Saved to all_dataset_queries.json")


if __name__ == "__main__":
    main()