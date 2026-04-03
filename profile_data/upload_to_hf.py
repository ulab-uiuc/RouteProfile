"""Upload each profile_data/ JSON to its own ulab-ai/RouteProfile-* dataset repo."""
import os
from huggingface_hub import HfApi

api = HfApi()

LOCAL_DIR = os.path.join(os.path.dirname(__file__), "profile_data")

FILE_REPO_MAP = {
    "candidate_models.json":      "ulab-ai/RouteProfile-candidate-models",
    "domain_feature.json":        "ulab-ai/RouteProfile-domain-feature",
    "domain_task_map.json":       "ulab-ai/RouteProfile-domain-task-map",
    "model_family_feature.json":  "ulab-ai/RouteProfile-model-family-feature",
    "model_feature_newllm.json":  "ulab-ai/RouteProfile-model-feature-newllm",
    "model_feature_standard.json":"ulab-ai/RouteProfile-model-feature-standard",
    "task_feature.json":          "ulab-ai/RouteProfile-task-feature",
    "task_queries_newllm.json":   "ulab-ai/RouteProfile-task-queries-newllm",
    "task_queries_standard.json": "ulab-ai/RouteProfile-task-queries-standard",
}

for fname, repo_id in FILE_REPO_MAP.items():
    local_path = os.path.join(LOCAL_DIR, fname)
    print(f"Creating repo {repo_id} ...")
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    print(f"Uploading {fname} -> {repo_id} ...")
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=fname,
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"  Done.\n")

print("All files uploaded successfully.")
