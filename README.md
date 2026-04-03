# RouteProfile: Elucidating the Design Space of LLM Profiles for Routing

As the large language model (LLM) ecosystem expands, individual models exhibit varying capabilities across queries, benchmarks, and domains, motivating the development of LLM routing. While prior work has largely focused on router mechanism design, **LLM profiles**—which capture model capabilities—remain underexplored.

**RouteProfile** treats LLM profiling as a structured information integration problem over heterogeneous interaction histories and develops a general design space of LLM profiles along four key dimensions: *organizational form*, *representation type*, *aggregation depth*, and *learning configuration*. Through systematic evaluation across three representative routers under both standard and new-LLM generalization settings, we show that structured profiles consistently outperform flat ones, query-level signals are more reliable than coarse domain-level signals, and generalization to newly introduced models benefits most from structured profiles under trainable configurations.

---

## Installation

```bash
pip install routeprofile
```

For Text-GNN profiles (requires vLLM):

```bash
pip install "routeprofile[text-gnn]"
```

Install from source (editable):

```bash
git clone https://github.com/your-org/RouteProfile.git
cd RouteProfile
pip install -e .
```

---

## Pipeline Overview

```
Step 1: Data Collection   →  profile_data/                        (manual / provided)
Step 2: Build Data Graph  →  results/result_data_graph/{mode}/
Step 3: Build Profile     →  results/model_profile_result/{mode}/
Step 4: Route & Evaluate  →  results/routing_result/{mode}/
```

Two routing settings:

| Mode       | Description                                         |
|------------|-----------------------------------------------------|
| `standard` | Standard routing with a known set of candidate LLMs |
| `newllm`   | Generalisation to newly introduced, unseen LLMs     |

---

## Python API

All functions are importable directly from `routeprofile`:

```python
import routeprofile
print(routeprofile.__version__)  # "0.1.0"
```

### Step 2 — Build Data Graphs

```python
from routeprofile import (
    build_task_graph,
    build_query_graph,
    build_query_task_graph,
    build_task_domain_graph,
    build_query_task_domain_graph,
)

# Uses default profile_data/ inputs; outputs to results/result_data_graph/standard/
build_task_graph(mode="standard")

# Override any input/output path
build_query_task_domain_graph(
    mode="standard",
    json="profile_data/model_feature_standard.json",
    arch="profile_data/model_family_feature.json",
    dataset="profile_data/task_feature.json",
    query="profile_data/task_queries_standard.json",
    domain_map="profile_data/domain_task_map.json",
    domain_feat="profile_data/domain_feature.json",
    save="results/result_data_graph/standard/query_task_domain_graph_full.pt",
)
```

### Step 3a — Training-Free Profiles

```python
from routeprofile import (
    build_flat_profile,
    build_emb_gnn_profile,
    build_index_profile,
    build_text_gnn_profile,
)

# Flat: Longformer encoding of model text + sampled neighbours
build_flat_profile(mode="standard")
# → results/model_profile_result/standard/flat.npz

# Index: random vector baseline (no text or graph)
build_index_profile(mode="standard")
# → results/model_profile_result/standard/index.npz

# Emb-GNN: K-hop neighbourhood propagation (training-free)
build_emb_gnn_profile(
    mode="standard",
    graph="results/result_data_graph/standard/task_graph_full.pt",
    K=2,
    norm="sym",   # "sym" | "rw" | "none"
    save="results/model_profile_result/standard/emb_gnn.npz",
)

# Text-GNN: LLM-based text aggregation per hop (requires vLLM)
build_text_gnn_profile(
    mode="standard",
    graph="results/result_data_graph/standard/query_task_domain_graph_full.pt",
    K=1,
    model="Qwen/Qwen2.5-7B-Instruct",
    tp=1,                        # tensor parallel size (number of GPUs)
    gpu_memory_utilization=0.6,  # fraction of GPU memory for vLLM
    keep=[],                     # [] = save all models; None = TARGET_MODELS only
    emb_save="results/model_profile_result/standard/text_gnn.npz",
)
```

### Step 3b — Trainable GNN Profile (HANConv)

```python
from routeprofile import build_trainable_gnn_profile

build_trainable_gnn_profile(
    mode="standard",
    graph="results/result_data_graph/standard/task_graph_full.pt",
    hidden_dim=256,
    out_dim=128,
    epochs=100,
    save_emb="results/model_profile_result/standard/trainable_gnn.npz",
    save_ckpt="results/trained_trainable_gnn/standard/pretrain_ckpt.pt",
)
```

### Step 4 — Routing Evaluation

```python
from routeprofile import call_simrouter, call_mlprouter, call_graphrouter

# SimRouter: training-free cosine similarity routing
call_simrouter(
    model_profile_path="results/model_profile_result/standard/flat.npz",
    routing_data_path="route_data/routing_test_data.json",
    output_path="results/routing_result/standard/SimRouter_results.json",
)

# MLPRouter: pairwise-ranking MLP
call_mlprouter(
    model_profile_path="results/model_profile_result/standard/emb_gnn.npz",
    training_data_path="route_data/pairwise_training_data_standard.json",
    testing_data_path="route_data/routing_test_data.json",
    output_path="results/routing_result/standard/MLPRouter_results.json",
    save_ckpt="results/trained_MLPRouter/standard/mlp_router_ckpt.pt",
    epochs=50,
)

# GraphRouter: bipartite GAT
call_graphrouter(
    model_profile_path="results/model_profile_result/standard/trainable_gnn.npz",
    training_data_path="route_data/pairwise_training_data_standard.json",
    testing_data_path="route_data/routing_test_data.json",
    output_path="results/routing_result/standard/GraphRouter_results.json",
    save_ckpt="results/trained_GraphRouter/standard/graphrouter_ckpt.pt",
    epochs=50,
)
```

You can also import the router classes directly:

```python
from routeprofile import SimRouter, MLPRouter, GraphRouter
```

---

## CLI

After installation every step is available as a command-line tool:

```bash
# Step 2: Build graphs (outputs to results/result_data_graph/{mode}/)
routeprofile-build-task-graph               --mode standard
routeprofile-build-query-graph              --mode standard
routeprofile-build-query-task-graph         --mode standard
routeprofile-build-task-domain-graph        --mode standard
routeprofile-build-query-task-domain-graph  --mode standard

# Step 3a: Training-free profiles (outputs to results/model_profile_result/{mode}/)
routeprofile-flat-profile      --mode standard
routeprofile-index-profile     --mode standard
routeprofile-emb-gnn-profile   --mode standard --K 2
routeprofile-trainable-gnn-profile --mode standard --epochs 100

# Step 4: Routing (outputs to results/routing_result/{mode}/)
routeprofile-sim-router \
    --model-profile-path results/model_profile_result/standard/flat.npz \
    --routing-data-path  route_data/routing_test_data.json

routeprofile-mlp-router \
    --model-profile-path  results/model_profile_result/standard/emb_gnn.npz \
    --training-data-path  route_data/pairwise_training_data_standard.json \
    --testing-data-path   route_data/routing_test_data.json \
    --save-ckpt           results/trained_MLPRouter/standard/mlp_router_ckpt.pt

routeprofile-graph-router \
    --model-profile-path  results/model_profile_result/standard/trainable_gnn.npz \
    --training-data-path  route_data/pairwise_training_data_standard.json \
    --testing-data-path   route_data/routing_test_data.json \
    --save-ckpt           results/trained_GraphRouter/standard/graphrouter_ckpt.pt
```

All commands accept `--help` for full usage.

---

## Shell Scripts (Batch Runs)

```bash
# Build all graphs (standard mode)
bash routeprofile/scripts/step2_build_data_graph.sh standard

# All training-free profiles
bash routeprofile/scripts/step3a_training_free_profile.sh standard all

# Text-GNN (requires vLLM + GPU)
bash routeprofile/scripts/step3a_training_free_profile.sh standard text_gnn

# Trainable GNN
bash routeprofile/scripts/step3b_trainable_profile.sh standard

# Routing evaluation
bash routeprofile/scripts/step4_routing_evaluation.sh standard sim flat.npz
bash routeprofile/scripts/step4_routing_evaluation.sh standard all flat.npz
```

---

## Profile Methods

| Method      | File                  | Org. form  | Repr. type  | Agg. depth | Learning |
|-------------|-----------------------|------------|-------------|------------|----------|
| `flat`      | `flat.npz`            | Flat       | Text        | Shallow    | None     |
| `index`     | `index.npz`           | Flat       | Random      | None       | None     |
| `emb_gnn`   | `emb_gnn.npz`         | Structured | Embedding   | Multi-hop  | None     |
| `text_gnn`  | `text_gnn.npz`        | Structured | Text + LLM  | Multi-hop  | None     |
| `trainable` | `trainable_gnn.npz`   | Structured | Embedding   | Multi-hop  | Self-sup |

## Router Methods

| Router       | Type          | Description                                          |
|--------------|---------------|------------------------------------------------------|
| `SimRouter`  | Training-free | Cosine similarity between query and model embeddings |
| `MLPRouter`  | Trainable     | Pairwise ranking loss; query + model encoders        |
| `GraphRouter`| Trainable     | Bipartite GAT with edge prediction (BCE loss)        |

---

## Directory Structure

```
RouteProfile/
├── profile_data/                        # Input data (read-only)
│   ├── model_feature_standard.json          # Model metadata (standard routing)
│   ├── model_feature_newllm.json            # Model metadata (newllm routing)
│   ├── model_family_feature.json            # Architecture family descriptions
│   ├── task_queries_standard.json           # Queries per benchmark (standard)
│   ├── task_queries_newllm.json             # Queries per benchmark (newllm)
│   ├── task_feature.json                    # Benchmark task descriptions
│   ├── domain_feature.json                  # Task domain descriptions
│   ├── domain_task_map.json                 # Domain → benchmark mapping
│   └── candidate_models.json               # Candidate LLM metadata
│
├── route_data/                          # Pre-computed routing data
│   ├── routing_test_data.json               # Test queries with model responses
│   ├── pairwise_training_data_standard.json # Pairwise training data (standard)
│   └── pairwise_training_data_newllm.json   # Pairwise training data (newllm)
│
├── routeprofile/                        # Library source
│   ├── build_data_graph/                    # Step 2: graph construction
│   ├── get_model_profile/
│   │   ├── training_free/                   # flat, index, emb_gnn, text_gnn
│   │   └── trainable/                       # HANConv self-supervised
│   ├── routing_evaluation/                  # SimRouter, MLPRouter, GraphRouter
│   └── scripts/                             # Shell scripts for batch runs
│
├── results/                             # All generated outputs (git ignored)
│   ├── result_data_graph/{standard,newllm}/     # Built graphs (.pt)
│   ├── model_profile_result/{standard,newllm}/  # Model profiles (.npz)
│   ├── routing_result/{standard,newllm}/        # Routing evaluation results (.json)
│   ├── trained_trainable_gnn/{standard,newllm}/ # HANConv checkpoints
│   ├── trained_MLPRouter/{standard,newllm}/     # MLP router checkpoints
│   └── trained_GraphRouter/{standard,newllm}/   # Graph router checkpoints
│
├── tests/                               # pytest test suite
└── pyproject.toml
```

---

## Data Formats

### `profile_data/model_feature_{standard|newllm}.json`
Main model metadata. Primary input to all graph builders.
```json
{
  "model-name": {
    "size": "7B",
    "feature": "Natural language description of the model...",
    "architecture": "Qwen2ForCausalLM",
    "detailed_scores": {
      "ifeval": 75.85, "bbh": 53.94, "math": 50.0,
      "gpqa": 29.11, "musr": 40.2, "mmlu_pro": 42.87
    },
    "parameters": 7.616,
    "input_price": 0.2,
    "output_price": 0.2,
    "model": "qwen/qwen2.5-7b-instruct",
    "service": "NVIDIA",
    "api_endpoint": "https://integrate.api.nvidia.com/v1",
    "average_score": 35.2
  }
}
```

### `profile_data/model_family_feature.json`
Architecture family descriptions used as architecture node features.
```json
{
  "Qwen2ForCausalLM": "A family of decoder-only Transformer-based large language models developed by Alibaba Cloud...",
  "LlamaForCausalLM": "A family of autoregressive large language models developed by Meta AI..."
}
```

### `profile_data/task_feature.json`
Natural language description of each benchmark task.
```json
{
  "ifeval": "IFEval (Instruction-Following Evaluation) is a benchmark designed to evaluate the ability of large language models to follow explicit natural language instructions...",
  "bbh":    "BBH (BIG-Bench Hard) is a challenging subset of the BIG-Bench benchmark..."
}
```

### `profile_data/domain_task_map.json`
Maps broad task domains to specific benchmarks.
```json
{
  "knowledge": ["mmlu", "mmlu_pro", "C-Eval", "AGIEval English", "SQuAD", "gpqa"],
  "reasoning": ["bbh", "TheoremQA", "WinoGrande"],
  "math":      ["math", "gsm8k", "TheoremQA"],
  "coding":    ["human_eval", "mbpp"]
}
```

### `profile_data/domain_feature.json`
Natural language description of each task domain.
```json
{
  "knowledge": "Knowledge tasks test factual recall and information retrieval...",
  "reasoning": "Reasoning tasks require multi-step logical inference...",
  "math":      "Math tasks evaluate quantitative and symbolic problem solving..."
}
```

### `profile_data/candidate_models.json`
Candidate model metadata including API endpoints and aggregate scores.
```json
{
  "qwen2.5-7b-instruct": {
    "size": "7B",
    "feature": "Qwen2.5-7B-Instruct represents an upgraded version...",
    "input_price": 0.2,
    "output_price": 0.2,
    "model": "qwen/qwen2.5-7b-instruct",
    "service": "NVIDIA",
    "api_endpoint": "https://integrate.api.nvidia.com/v1",
    "average_score": 35.2,
    "detailed_scores": { "ifeval": 75.85, "bbh": 53.94 },
    "parameters": 7.616,
    "architecture": "Qwen2ForCausalLM"
  }
}
```

### `profile_data/task_queries_{standard|newllm}.json`
Per-benchmark query lists used to build query nodes.
```json
{
  "ifeval": ["Instruction 1...", "Instruction 2...", ...],
  "bbh":    ["Question 1...",   "Question 2...",   ...]
}
```

### `route_data/routing_test_data.json`
Pre-computed model responses for test queries.
```json
[
  {
    "task_name": "ifeval",
    "query": "Follow these instructions...",
    "ground_truth": "A",
    "metric": "em_mc",
    "choices": "{'text': ['A', 'B', 'C', 'D'], 'labels': ['A', 'B', 'C', 'D']}",
    "model_performance": {
      "qwen2.5-7b-instruct": { "response": "A", "task_performance": 1.0, "success": true }
    }
  }
]
```

### `route_data/pairwise_training_data_{standard|newllm}.json`
Pairwise training data for MLPRouter and GraphRouter. Each entry records which model outperforms which on a given query.
```json
{
  "task_data_count": {
    "agentverse-logicgrid": 1352,
    "gsm8k": 741
  },
  "pairwise_data": [
    {
      "task_name": "agentverse-logicgrid",
      "query": "Q: There are 4 houses...",
      "ground_truth": "B",
      "metric": "em_mc",
      "choices": "{'text': ['1', '2', '3', '4'], 'labels': ['A', 'B', 'C', 'D']}",
      "task_id": null,
      "better_model": "mistral-small-24b-instruct-2501-bf16",
      "worse_model":  "mixtral-8x22b-instruct-v0.1"
    }
  ]
}
```

> **Note:** Use `pairwise_training_data_{mode}.json` as `training_data_path` for MLPRouter and GraphRouter. The `routing_test_data.json` is used for `testing_data_path`.

---

## Candidate Models

The default set of 8 candidate models:

| Model                                   | Size  | Architecture       |
|-----------------------------------------|-------|--------------------|
| `qwen2.5-7b-instruct`                   | 7B    | Qwen2ForCausalLM   |
| `gemma-2-9b-it`                         | 9B    | Gemma2ForCausalLM  |
| `llama-3.1-8b-instruct`                 | 8B    | LlamaForCausalLM   |
| `mixtral-8x7b-instruct-v0.1`            | 46.7B | MixtralForCausalLM |
| `mixtral-8x22b-instruct-v0.1`           | 141B  | MixtralForCausalLM |
| `llama-3.2-3b-instruct`                 | 3B    | LlamaForCausalLM   |
| `mistral-small-24b-instruct-2501-bf16`  | 24B   | MistralForCausalLM |
| `llama-3.3-70b-instruct`                | 70B   | LlamaForCausalLM   |

---

## Citation

If you use RouteProfile in your research, please cite:

```bibtex
@article{routeprofile2025,
  title={RouteProfile: Elucidating the Design Space of LLM Profiles for Routing},
  year={2025}
}
```
