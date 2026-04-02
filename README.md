# RouteProfile: Elucidating the Design Space of LLM Profiles for Routing

RouteProfile is a framework for building **model profiles** from heterogeneous graphs and using them for LLM routing. Given a query, a router selects the most suitable LLM from a pool of candidates based on their profiles.

## Overview

The pipeline consists of four steps:

```
Step 1: Data Collection  →  profile_data/   (manual / uploaded)
Step 2: Build Data Graph →  profile_data/result_data_graph/{mode}/
Step 3: Build Model Profile → routeprofile/model_profile_result/{mode}/
Step 4: Routing Evaluation  → routeprofile/routing_result/{mode}/
```

Two routing settings are supported:

| Mode       | Description                                    |
|------------|------------------------------------------------|
| `standard` | Standard LLM routing with known candidate models |
| `newllm`   | Routing scenario with new/unseen LLMs          |

---

## Quick Start

```bash
# Clone and install dependencies
pip install -r requirements.txt

# Build graphs (standard mode)
bash routeprofile/scripts/step2_build_data_graph.sh standard

# Generate training-free profiles
bash routeprofile/scripts/step3a_training_free_profile.sh standard all

# OR: Train a GNN-based profile
bash routeprofile/scripts/step3b_trainable_profile.sh standard

# Evaluate routing
bash routeprofile/scripts/step4_routing_evaluation.sh standard sim flat.npz
bash routeprofile/scripts/step4_routing_evaluation.sh standard all flat.npz
```

---

## Directory Structure

```
RouteProfile/
├── profile_data/                  # Input data for graph construction
│   ├── model_feature_standard.json    # Model metadata (standard routing)
│   ├── model_feature_newllm.json      # Model metadata (newllm routing)
│   ├── task_queries_standard.json     # Queries per benchmark (standard)
│   ├── task_queries_newllm.json       # Queries per benchmark (newllm)
│   ├── task_feature.json              # Benchmark task descriptions (shared)
│   ├── domain_feature.json            # Task domain descriptions (shared)
│   ├── domain_task_map.json           # Domain → benchmark mapping (shared)
│   ├── candidate_models.json          # Candidate LLM metadata (shared)
│   └── result_data_graph/
│       ├── standard/                  # Built graphs for standard mode
│       └── newllm/                    # Built graphs for newllm mode
│
├── route_data/                    # Pre-computed routing data (all models)
│   ├── routing_train_data.json        # Training queries with model responses
│   └── routing_test_data.json         # Test queries with model responses
│
└── routeprofile/
    ├── build_data_graph/          # Step 2: graph construction scripts
    ├── get_model_profile/
    │   ├── training_free/         # Step 3a: flat, index, emb_gnn, text_gnn
    │   └── trainable/             # Step 3b: HANConv self-supervised training
    │       └── trained_gnn/
    │           ├── standard/      # Saved HANConv checkpoints (standard)
    │           └── newllm/        # Saved HANConv checkpoints (newllm)
    ├── model_profile_result/      # Output model profiles (.npz)
    │   ├── standard/
    │   └── newllm/
    ├── routing_evaluation/        # Step 4: SimRouter, MLPRouter, GraphRouter
    │   ├── trained_MLPRouter/     # Saved MLP router checkpoints
    │   └── trained_GraphRouter/   # Saved Graph router checkpoints
    ├── routing_result/            # Evaluation outputs (.json)
    │   ├── standard/
    │   └── newllm/
    └── scripts/                   # Shell scripts for each step
```

---

## Step 1: Data Collection

Data is manually prepared and placed in `profile_data/`. The key files and their formats are:

### `model_feature_{standard|newllm}.json`
Main model metadata. Used as the primary input to all graph builders.
```json
{
  "model-name": {
    "size": "7B",
    "feature": "Natural language description of the model...",
    "architecture": "Qwen2ForCausalLM",
    "detailed_scores": {
      "ifeval": 75.85,
      "bbh": 53.94,
      "math": 50.0,
      "gpqa": 29.11,
      "musr": 40.2,
      "mmlu_pro": 42.87
    },
    "parameters": 7.616,
    "input_price": 0.2,
    "output_price": 0.2
  }
}
```

### `task_queries_{standard|newllm}.json`
Per-benchmark query lists. Used to build query nodes in the graph.
```json
{
  "ifeval": ["Instruction 1...", "Instruction 2...", ...],
  "bbh":    ["Question 1...",    "Question 2...",    ...]
}
```

### `task_feature.json`
Natural language description of each benchmark task (shared across modes).
```json
{
  "ifeval": "IFEval is a benchmark designed to evaluate...",
  "bbh":    "BBH (BIG-Bench Hard) is a challenging subset..."
}
```

### `domain_task_map.json`
Maps broad task domains to specific benchmarks (shared).
```json
{
  "knowledge": ["mmlu", "mmlu_pro", "C-Eval"],
  "reasoning": ["bbh", "TheoremQA"],
  "math":      ["math", "TheoremQA"]
}
```

### `domain_feature.json`
Natural language description of each task domain (shared).
```json
{
  "knowledge": "Knowledge tasks test factual recall...",
  "reasoning": "Reasoning tasks require multi-step inference..."
}
```

### `route_data/routing_train_data.json` and `routing_test_data.json`
Pre-computed model responses for each query, used for router training and evaluation.
```json
[
  {
    "task_name": "ifeval",
    "query": "Follow these instructions...",
    "ground_truth": "A",
    "metric": "em_mc",
    "choices": "{'text': ['A', 'B', 'C', 'D'], 'labels': ['A', 'B', 'C', 'D']}",
    "model_performance": {
      "qwen2.5-7b-instruct": {
        "response": "A",
        "task_performance": 1.0,
        "success": true
      }
    }
  }
]
```

---

## Step 2: Build Data Graphs

Constructs PyTorch Geometric heterogeneous graphs from the profile data.

```bash
# Build all 5 graph variants for standard mode
bash routeprofile/scripts/step2_build_data_graph.sh standard

# Build for newllm mode
bash routeprofile/scripts/step2_build_data_graph.sh newllm

# Build both modes
bash routeprofile/scripts/step2_build_data_graph.sh both
```

Five graph types are built per mode:

| Graph file                         | Node types                              |
|------------------------------------|-----------------------------------------|
| `task_graph_full.pt`               | architecture, model, dataset            |
| `query_graph_full.pt`              | architecture, model, query              |
| `query_task_graph_full.pt`         | architecture, model, dataset, query     |
| `task_domain_graph_full.pt`        | architecture, model, dataset, domain    |
| `query_task_domain_graph_full.pt`  | architecture, model, dataset, domain, query |

To inspect a built graph:
```bash
python routeprofile/build_data_graph/print_graph.py \
    --graph profile_data/result_data_graph/standard/query_task_domain_graph_full.pt
```

---

## Step 3a: Training-Free Model Profiles

Generates model embeddings without training a GNN.

```bash
# All methods (flat, index, emb_gnn) for standard mode
bash routeprofile/scripts/step3a_training_free_profile.sh standard all

# Single method
bash routeprofile/scripts/step3a_training_free_profile.sh standard emb_gnn

# text_gnn (requires vLLM + GPU)
bash routeprofile/scripts/step3a_training_free_profile.sh standard text_gnn
```

| Method    | File         | Description                                                |
|-----------|--------------|------------------------------------------------------------|
| `flat`    | `flat.npz`   | Concatenates model text + random-sampled neighbours → Longformer |
| `index`   | `index.npz`  | Random vector baseline (no graph, no text encoding)        |
| `emb_gnn` | `emb_gnn.npz`| K-hop neighbourhood propagation with degree normalisation  |
| `text_gnn`| `text_gnn.npz`| LLM-based text aggregation per hop (requires vLLM)        |

---

## Step 3b: Trainable GNN Profile

Trains a HANConv encoder via self-supervised masked feature reconstruction.

```bash
# Standard mode with default hyperparameters
bash routeprofile/scripts/step3b_trainable_profile.sh standard

# Custom hyperparameters
bash routeprofile/scripts/step3b_trainable_profile.sh standard \
    --epochs 200 --hidden-dim 512 --node-mask-rate 0.4

# newllm mode
bash routeprofile/scripts/step3b_trainable_profile.sh newllm
```

Output: `routeprofile/model_profile_result/{mode}/trainable_gnn.npz`

---

## Step 4: Routing Evaluation

Evaluates routing performance using three router types.

```bash
# SimRouter (no training, cosine similarity)
bash routeprofile/scripts/step4_routing_evaluation.sh standard sim flat.npz

# MLPRouter (pairwise ranking training)
bash routeprofile/scripts/step4_routing_evaluation.sh standard mlp emb_gnn.npz

# GraphRouter (bipartite GAT training)
bash routeprofile/scripts/step4_routing_evaluation.sh standard graph trainable_gnn.npz

# Run all three routers
bash routeprofile/scripts/step4_routing_evaluation.sh standard all flat.npz
```

| Router       | Type       | Description                                                |
|--------------|------------|------------------------------------------------------------|
| `SimRouter`  | Training-free | Selects model with highest cosine similarity to query   |
| `MLPRouter`  | Trainable  | Pairwise ranking loss; query + model MLPs                  |
| `GraphRouter`| Trainable  | Bipartite GAT + edge prediction (BCE loss)                 |

Results are saved to `routeprofile/routing_result/{mode}/`.

---

## Candidate Models

The default set of 8 candidate models:

| Model                                | Size  | Architecture      |
|--------------------------------------|-------|-------------------|
| `qwen2.5-7b-instruct`                | 7B    | Qwen2ForCausalLM  |
| `gemma-2-9b-it`                      | 9B    | Gemma2ForCausalLM |
| `llama-3.1-8b-instruct`              | 8B    | LlamaForCausalLM  |
| `mixtral-8x7b-instruct-v0.1`         | 46.7B | MistralForCausalLM|
| `mixtral-8x22b-instruct-v0.1`        | 141B  | MistralForCausalLM|
| `llama-3.2-3b-instruct`              | 3B    | LlamaForCausalLM  |
| `mistral-small-24b-instruct-2501-bf16` | 24B | MistralForCausalLM|
| `llama-3.3-70b-instruct`             | 70B   | LlamaForCausalLM  |

---

## Citation

If you use RouteProfile in your research, please cite:

```bibtex
@article{routeprofile2024,
  title={RouteProfile: Elucidating the Design Space of LLM Profiles for Routing},
  year={2024}
}
```
