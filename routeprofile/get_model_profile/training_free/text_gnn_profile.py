"""
Text-GNN: LLM-based neighbourhood aggregation for heterogeneous graphs.
 
Each propagation hop replaces the numeric AX aggregation with a language
model call.  For every model node the LLM receives a structured prompt that
contains:
  - The node's own current node_feature_text
  - Each neighbour's node_feature_text, labelled by neighbour type
  - benchmark scores on the connecting edge (model <-> dataset edges only)
 
The LLM produces a new summary text that fuses all of this context.
That summary becomes the node's updated node_feature_text for the next hop,
and is also encoded by BERT into a dense embedding.
 
After K hops every model node has:
  - .node_feature_text   – the LLM-generated summary from the last hop
                           (or the original node_feature_text if K=0)
  - .x                   – BERT CLS embedding of that text  [768]
 
K=0 special case: no LLM is loaded; the original node_feature_text is encoded
directly by BERT.  This is useful as a text-only baseline.
 
Pipeline
--------
    data    = torch.load("llm_hetero_graph.pt", weights_only=False)
    output  = text_propagate(data, K=2, vllm_model="meta-llama/Llama-3.1-8B-Instruct")
 
    # K=0 baseline (no LLM needed):
    output  = text_propagate(data, K=0)
 
    # output["model_texts"]       – dict { model_name: text  (final hop, or raw if K=0) }
    # output["model_embeddings"]  – dict { model_name: np.ndarray [768] }
    # output["hop_texts"]         – dict { hop: { model_name: text } }  (all hops)
 
Install dependencies:
    pip install torch torch_geometric transformers vllm numpy
"""
 
import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from torch import Tensor
from torch_geometric.data import HeteroData
from vllm import LLM, SamplingParams
from llmrouter.utils import get_longformer_embedding
 
 
# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_VLLM_MODEL    = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 500
 
 
# ── Data containers ────────────────────────────────────────────────────────────
 
@dataclass
class TextGNNOutput:
    """
    All outputs produced by text_propagate().
 
    Attributes:
        model_texts      : { model_name: text }  from the final hop
                           (original node_feature_text if K=0)
        model_embeddings : { model_name: np.ndarray [768] }  BERT embeddings
        hop_texts        : { hop_index: { model_name: text } }
                           hop 0 = original node_feature_text (always present)
                           hop 1..K = LLM-generated summaries
                           K=0: only hop 0 is present
    """
    model_texts:      dict[str, str]
    model_embeddings: dict[str, np.ndarray]
    hop_texts:        dict[int, dict[str, str]]
 
 
# ── Prompt construction ────────────────────────────────────────────────────────
 
def _render_neighbour_block(nb: dict, indent: str = "  ") -> str:
    """
    Render a single neighbour node (and its children, if any) as indented text.
 
    Neighbour dict fields:
      "type"     : str              – node type
      "text"     : str              – node_feature_text
      "score"    : float | None     – benchmark score (dataset only)
      "domains"  : list[str]        – parent domain descriptions (dataset only)
      "children" : list[dict]       – 2-hop neighbours of this node (may be absent)
 
    Returns a multi-line string ready for embedding in the prompt.
    """
    ntype = nb["type"]
    text  = nb["text"]
    score = nb.get("score")
    children: list[dict] = nb.get("children", [])
 
    # ── header line ───────────────────────────────────────────────────────────
    if ntype == "dataset":
        score_tag = f" (score: {score:.2f})" if score is not None else ""
        domains   = nb.get("domains", [])
        dom_tag   = f" [domain: {', '.join(d[:60] for d in domains)}]" if domains else ""
        header    = f"{indent}[DATASET{score_tag}{dom_tag}] {text}"
    elif ntype == "architecture":
        header = f"{indent}[ARCHITECTURE] {text}"
    elif ntype == "query":
        header = f"{indent}[QUERY] {text}"
    elif ntype == "domain":
        header = f"{indent}[DOMAIN] {text}"
    else:
        header = f"{indent}[{ntype.upper()}] {text}"
 
    if not children:
        return header
 
    # ── children block (2-hop) ────────────────────────────────────────────────
    child_indent = indent + "  "
    child_lines  = [header, f"{indent}  └─ related context:"]
    for ch in children:
        child_lines.append(_render_neighbour_block(ch, indent=child_indent))
    return "\n".join(child_lines)
 
 
def _build_prompt(
    model_name:  str,
    self_text:   str,
    neighbours:  list[dict],   # 1-hop neighbours; each may carry "children" for 2-hop
    hop_depth:   int = 1,
) -> str:
    """
    Build the LLM prompt for one model node's aggregation step.
 
    Each neighbour dict (1-hop) contains:
        "type"     : str         – "architecture", "dataset", "query", or "domain"
        "text"     : str         – node_feature_text
        "score"    : float|None  – benchmark score (dataset edges only)
        "domains"  : list[str]   – domain descriptions this dataset belongs to
        "children" : list[dict]  – top-k 2-hop neighbours of this node
                                   (present only when hop_depth >= 2)
 
    The prompt structure mirrors the graph topology:
      - 1-hop: flat list, each neighbour labelled by type
      - 2-hop: tree — each 1-hop node is followed by its top-k 2-hop neighbours
               indented beneath it, making the edge relationships explicit
 
    Args:
        model_name : human-readable model key
        self_text  : the model node's current node_feature_text
        neighbours : list of 1-hop neighbour dicts
        hop_depth  : 1 or 2; controls whether children are rendered
 
    Returns:
        A fully formatted prompt string ready to pass to the LLM.
    """
    # separate by type for ordered section rendering
    arch_nbs:    list[dict] = []
    dataset_nbs: list[dict] = []
    query_nbs:   list[dict] = []
    other_nbs:   list[dict] = []
 
    for nb in neighbours:
        t = nb["type"]
        if t == "architecture":
            arch_nbs.append(nb)
        elif t == "dataset":
            dataset_nbs.append(nb)
        elif t == "query":
            query_nbs.append(nb)
        elif t == "domain":
            pass   # domain rendered inline with dataset via "domains" field
        else:
            other_nbs.append(nb)
 
    sections: list[str] = []
 
    # ── architectural family ───────────────────────────────────────────────────
    if arch_nbs:
        lines = [_render_neighbour_block(nb) for nb in arch_nbs]
        sections.append("### Architectural family\n" + "\n".join(lines))
 
    # ── benchmark performance (datasets, grouped by domain, with 2-hop context)
    if dataset_nbs:
        # group by domain
        domain_buckets: dict[str, list[dict]] = {}
        undomained:     list[dict]            = []
        for nb in dataset_nbs:
            doms = nb.get("domains", [])
            if doms:
                for d in doms:
                    domain_buckets.setdefault(d, []).append(nb)
            else:
                undomained.append(nb)
 
        perf_lines: list[str] = []
        for dom_text, nbs in domain_buckets.items():
            perf_lines.append(f"  [Domain] {dom_text[:120]}")
            for nb in nbs:
                perf_lines.append(_render_neighbour_block(nb, indent="    "))
        for nb in undomained:
            perf_lines.append(_render_neighbour_block(nb))
 
        sections.append(
            "### Benchmark performance by domain\n"
            "Datasets are grouped under their task domain. "
            "Indented items show additional context reachable via the graph.\n"
            + "\n".join(perf_lines)
        )
 
    # ── representative queries ─────────────────────────────────────────────────
    if query_nbs:
        lines = [_render_neighbour_block(nb) for nb in query_nbs]
        sections.append(
            "### Representative queries this model should handle\n"
            "Real user queries from the benchmarks, illustrating expected task types.\n"
            + "\n".join(lines)
        )
 
    # ── other node types ───────────────────────────────────────────────────────
    if other_nbs:
        lines = [_render_neighbour_block(nb) for nb in other_nbs]
        sections.append("### Additional context\n" + "\n".join(lines))
 
    context_block = "\n\n".join(sections) if sections else "(no neighbour context)"
 
    hop_note = (
        " The indented items under each node show further related context "
        "reachable at two hops from this model."
        if hop_depth >= 2 else ""
    )
 
    prompt = f"""You are writing a capability profile for a large language model to support intelligent model selection.
 
## Model being described
{self_text}
 
## Contextual information from the knowledge graph{hop_note}
{context_block}
 
## Task
Write a concise paragraph (3-5 sentences) that synthesises all of the above into a unified capability profile. Cover:
1. The model's architectural family and design characteristics.
2. Its performance across task domains based on the benchmark scores (and any additional graph context shown above).
3. The types of user queries it is best suited for, grounded in the representative queries above.
 
Output only the profile paragraph, with no headings, bullet points, or preamble."""
 
    return prompt
 
 
# ── Per-node-type prompt builders ────────────────────────────────────────────
# Each non-model node type gets its own prompt so the LLM knows what kind of
# summarisation is expected.  All builders take:
#   self_text  : the node's own current node_feature_text
#   neighbours : list of { type, text, score } dicts (1-hop only, no children)
# and return a prompt string ready for vLLM.
 
def _build_dataset_prompt(self_text: str, neighbours: list[dict]) -> str:
    """
    Prompt for a dataset node.
 
    Neighbours: model (with score), domain, query.
    Goal: describe what this benchmark evaluates, which models perform well/poorly,
    and what kinds of user questions it covers.
    """
    model_lines:  list[str] = []
    domain_lines: list[str] = []
    query_lines:  list[str] = []
 
    for nb in neighbours:
        t     = nb["type"]
        text  = nb["text"]
        score = nb.get("score")
        if t == "model":
            sc_tag = f" (score: {score:.2f})" if score is not None else ""
            model_lines.append(f"  - [MODEL{sc_tag}] {text}")
        elif t == "domain":
            domain_lines.append(f"  - [DOMAIN] {text}")
        elif t == "query":
            query_lines.append(f"  - [QUERY] {text}")
 
    sections: list[str] = []
    if domain_lines:
        sections.append("### Task domain this benchmark belongs to\n" + "\n".join(domain_lines))
    if model_lines:
        sections.append(
            "### Models evaluated on this benchmark (with scores)\n"
            + "\n".join(model_lines)
        )
    if query_lines:
        sections.append(
            "### Representative queries from this benchmark\n"
            + "\n".join(query_lines)
        )
    context = "\n\n".join(sections) if sections else "(no neighbour context)"
 
    return f"""You are writing a concise profile for an NLP benchmark dataset to support model-routing decisions.
 
## Benchmark being described
{self_text}
 
## Contextual information
{context}
 
## Task
Write a concise paragraph (2-4 sentences) that describes:
1. What capability or skill this benchmark evaluates.
2. Which models perform well or poorly on it, and at what score levels.
3. The types of user queries it is most relevant for.
 
Output only the profile paragraph, with no headings, bullet points, or preamble."""
 
 
def _build_domain_prompt(self_text: str, neighbours: list[dict]) -> str:
    """
    Prompt for a domain node.
 
    Neighbours: dataset.
    Goal: describe what capability area this domain covers and summarise the
    benchmark landscape within it.
    """
    dataset_lines: list[str] = []
    for nb in neighbours:
        if nb["type"] == "dataset":
            dataset_lines.append(f"  - [DATASET] {nb['text']}")
 
    context = (
        "### Benchmarks in this domain\n" + "\n".join(dataset_lines)
        if dataset_lines else "(no neighbour context)"
    )
 
    return f"""You are writing a concise profile for a task domain in NLP evaluation to support model-routing decisions.
 
## Domain being described
{self_text}
 
## Contextual information
{context}
 
## Task
Write a concise paragraph (2-4 sentences) that describes:
1. What this task domain covers and why it matters for model selection.
2. Which benchmarks represent it and what they collectively assess.
 
Output only the profile paragraph, with no headings, bullet points, or preamble."""
 
 
def _build_architecture_prompt(self_text: str, neighbours: list[dict]) -> str:
    """
    Prompt for an architecture node.
 
    Neighbours: model.
    Goal: characterise this architecture family based on what models use it
    and how those models perform.
    """
    model_lines: list[str] = []
    for nb in neighbours:
        if nb["type"] == "model":
            model_lines.append(f"  - [MODEL] {nb['text']}")
 
    context = (
        "### Models that use this architecture\n" + "\n".join(model_lines)
        if model_lines else "(no neighbour context)"
    )
 
    return f"""You are writing a concise profile for an LLM architecture family to support model-routing decisions.
 
## Architecture being described
{self_text}
 
## Contextual information
{context}
 
## Task
Write a concise paragraph (2-4 sentences) that describes:
1. The key design characteristics of this architecture.
2. The typical capability profile of models built on it, based on the models listed above.
 
Output only the profile paragraph, with no headings, bullet points, or preamble."""
 
 
def _build_prompt_for_node(
    ntype:      str,
    self_text:  str,
    neighbours: list[dict],
) -> str:
    """
    Dispatch to the appropriate per-type prompt builder.
 
    Args:
        ntype      : node type ("dataset", "domain", "architecture", ...)
        self_text  : current node_feature_text
        neighbours : 1-hop neighbour dicts (type, text, score)
 
    Returns:
        Prompt string.  Raises ValueError for "model" (use _build_prompt instead)
        and returns a generic prompt for unrecognised types.
    """
    if ntype == "dataset":
        return _build_dataset_prompt(self_text, neighbours)
    if ntype == "domain":
        return _build_domain_prompt(self_text, neighbours)
    if ntype == "architecture":
        return _build_architecture_prompt(self_text, neighbours)
    if ntype == "model":
        raise ValueError("Use _build_prompt() for model nodes, not _build_prompt_for_node().")
    # generic fallback for any future node types
    nb_lines = "\n".join(f"  - [{nb['type'].upper()}] {nb['text']}" for nb in neighbours) or "(none)"
    return f"""Summarise the following node in 2-3 sentences using the listed neighbours as context.
 
## Node text
{self_text}
 
## Neighbours
{nb_lines}
 
Output only the summary paragraph."""
 
 
# ── Generic 1-hop neighbour collector (for non-model node types) ──────────────
 
def _collect_neighbours_of(
    node_idx:         int,
    node_type:        str,
    data:             HeteroData,
    node_texts:       dict[str, list[str]],
    top_k:            int,
    embeddings_cache: dict[str, np.ndarray],
    exclude_types:    set[str] | None = None,
) -> list[dict]:
    """
    Collect top-k 1-hop neighbours of any node type.
 
    Used when updating dataset, domain, and architecture nodes.
    Unlike _collect_neighbours (which is model-specific), this function works
    for any pivot node type and always operates at depth=1.
 
    Similarity anchor: the pivot node's own embedding.
 
    Args:
        node_idx         : index of the pivot node
        node_type        : type string of the pivot node
        data             : HeteroData graph
        node_texts       : { ntype: [text, ...] }
        top_k            : max neighbours to keep per neighbour type
        embeddings_cache : pre-built { text: l2_normed_emb }
        exclude_types    : neighbour types to skip entirely
 
    Returns:
        List of neighbour dicts with keys: type, text, score.
    """
    if exclude_types is None:
        exclude_types = set()
 
    self_text = node_texts[node_type][node_idx]
    self_emb  = embeddings_cache.get(self_text)
 
    type_candidates: dict[str, list[tuple[str, float | None, float]]] = {}
 
    for (src_type, rel, dst_type), edge_store in data.edge_items():
        if "edge_index" not in edge_store:
            continue
        ei       = edge_store.edge_index
        has_attr = hasattr(edge_store, "edge_attr") and edge_store.edge_attr is not None
 
        # pivot is destination
        if dst_type == node_type:
            mask    = ei[1] == node_idx
            src_ids = ei[0][mask].tolist()
            scores  = (edge_store.edge_attr[mask, 0].tolist()
                       if has_attr else [None] * len(src_ids))
            for sid, sc in zip(src_ids, scores):
                if src_type in exclude_types:
                    continue
                nb_text = node_texts[src_type][sid]
                nb_emb  = embeddings_cache.get(nb_text)
                sim = float(np.dot(self_emb, nb_emb)) if (self_emb is not None and nb_emb is not None) else 0.0
                type_candidates.setdefault(src_type, []).append((nb_text, sc, sim))
 
        # pivot is source
        elif src_type == node_type:
            mask    = ei[0] == node_idx
            dst_ids = ei[1][mask].tolist()
            scores  = (edge_store.edge_attr[mask, 0].tolist()
                       if has_attr else [None] * len(dst_ids))
            for did, sc in zip(dst_ids, scores):
                if dst_type in exclude_types:
                    continue
                nb_text = node_texts[dst_type][did]
                nb_emb  = embeddings_cache.get(nb_text)
                sim = float(np.dot(self_emb, nb_emb)) if (self_emb is not None and nb_emb is not None) else 0.0
                type_candidates.setdefault(dst_type, []).append((nb_text, sc, sim))
 
    result: list[dict] = []
    for ntype, cands in type_candidates.items():
        # deduplicate by text
        seen: set[str] = set()
        unique = [c for c in cands if not (c[0] in seen or seen.add(c[0]))]
        if len(unique) > top_k:
            unique.sort(key=lambda x: x[2], reverse=True)
            unique = unique[:top_k]
        for (text, sc, _) in unique:
            result.append({"type": ntype, "text": text, "score": sc})
 
    return result
 
 
# ── Per-type update runner ────────────────────────────────────────────────────
 
def _update_node_type(
    ntype:            str,
    data:             HeteroData,
    node_texts:       dict[str, list[str]],   # read-only snapshot
    llm:              "LLM",
    sampling_params:  "SamplingParams",
    top_k:            int,
    embeddings_cache: dict[str, np.ndarray],
) -> list[str]:
    """
    Run one round of LLM aggregation for all nodes of a given non-model type.
 
    Reads from `node_texts` (a frozen snapshot) and returns the updated list.
 
    Args:
        ntype            : node type to update ("dataset", "domain", "architecture")
        data             : HeteroData graph
        node_texts       : frozen snapshot of all current node texts
        llm              : initialised vLLM instance
        sampling_params  : vLLM SamplingParams
        top_k            : max neighbours per type per node
        embeddings_cache : pre-built { text: l2_normed_emb }
 
    Returns:
        New list[str] of updated texts, same length and order as node_texts[ntype].
    """
    texts = node_texts[ntype]
    n     = len(texts)
 
    if n == 0:
        return list(texts)
 
    print(f"    [{ntype}] building {n} prompts ...")
    prompts: list[str] = []
    for node_idx in range(n):
        self_text  = texts[node_idx]
        neighbours = _collect_neighbours_of(
            node_idx=node_idx,
            node_type=ntype,
            data=data,
            node_texts=node_texts,
            top_k=top_k,
            embeddings_cache=embeddings_cache,
        )
        prompts.append(_build_prompt_for_node(ntype, self_text, neighbours))
 
    print(f"    [{ntype}] running vLLM ({n} prompts) ...")
    summaries = _run_vllm_batch(prompts, llm, sampling_params)
 
    for i, s in enumerate(summaries[:3]):   # preview first 3
        print(f"      [{ntype}][{i}]: {s[:80].replace(chr(10), ' ')}...")
 
    return summaries
 
 
# ── Cache builder for all-node update ────────────────────────────────────────
 
def _build_all_needed_texts(
    data:       HeteroData,
    node_texts: dict[str, list[str]],
) -> set[str]:
    """
    Collect every text that must be in the Longformer cache to support the
    all-node-type update in one hop.
 
    Includes: every non-query node's own text, plus every 1-hop neighbour's
    text reachable from a non-query node.  query texts are excluded because
    query nodes are never updated and their texts are not used as self-anchors.
    """
    needed: set[str] = set()
 
    for ntype, texts in node_texts.items():
        if ntype == "query":
            continue
        for t in texts:
            needed.add(t)
 
    for (src_type, rel, dst_type), edge_store in data.edge_items():
        if "edge_index" not in edge_store:
            continue
        ei = edge_store.edge_index
        if src_type != "query":
            for idx in ei[0].tolist():
                needed.add(node_texts[src_type][idx])
        if dst_type != "query":
            for idx in ei[1].tolist():
                needed.add(node_texts[dst_type][idx])
 
    return needed
 
 
# ── One-hop all-node propagation ──────────────────────────────────────────────
 
def _propagate_all_nodes_one_hop(
    data:            HeteroData,
    node_texts:      dict[str, list[str]],
    model_names:     list[str],
    llm:             "LLM",
    sampling_params: "SamplingParams",
    hop:             int,
    top_k:           int,
    update_types:    list[str],
) -> dict[str, list[str]]:
    """
    Execute one round of parallel snapshot-style LLM aggregation for all
    updatable node types.
 
    All node types read from the same frozen snapshot taken at the start of
    this hop, so no type benefits from updates applied to other types in the
    same hop.  Updates are collected and applied together at the end.
 
    Node types updated (all reading the same snapshot):
      dataset, domain, architecture, model
    query nodes are intentionally excluded — their raw text is preserved.
 
    Args:
        data            : HeteroData graph
        node_texts      : current node texts (will be snapshot-frozen)
        model_names     : ordered list of model node name strings
        llm             : initialised vLLM instance
        sampling_params : vLLM SamplingParams
        hop             : current hop index (1-based, for logging)
        top_k           : max neighbours per type per node
        update_types    : list of node types to update this hop
 
    Returns:
        New node_texts dict with all updatable types replaced by LLM outputs.
    """
    print(f"\n  Building Longformer cache for hop {hop} ...")
    needed    = _build_all_needed_texts(data, node_texts)
    emb_cache = _build_longformer_cache(list(needed), batch_size=8)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  Cache: {len(emb_cache)} unique texts encoded.")
 
    # freeze snapshot — all types read from this, not from each other's updates
    snapshot = {ntype: list(texts) for ntype, texts in node_texts.items()}
    updates:  dict[str, list[str]] = {}
 
    for ntype in update_types:
        if ntype not in snapshot:
            continue
 
        if ntype == "model":
            # model uses the existing _collect_neighbours + _build_prompt pipeline
            n_models = len(model_names)
            print(f"\n    [model] building {n_models} prompts ...")
            prompts: list[str] = []
            for model_idx, model_name in enumerate(model_names):
                self_text  = snapshot["model"][model_idx]
                neighbours = _collect_neighbours(
                    model_idx, data, snapshot,
                    top_k=top_k, hop_depth=1,
                    embeddings_cache=emb_cache,
                )
                prompts.append(_build_prompt(model_name, self_text, neighbours, hop_depth=1))
            print(f"    [model] running vLLM ({n_models} prompts) ...")
            summaries = _run_vllm_batch(prompts, llm, sampling_params)
            for i, (name, s) in enumerate(zip(model_names, summaries)):
                print(f"      [model][{i}] {name}: {s[:80].replace(chr(10), ' ')}...")
            updates["model"] = summaries
        else:
            updates[ntype] = _update_node_type(
                ntype=ntype,
                data=data,
                node_texts=snapshot,
                llm=llm,
                sampling_params=sampling_params,
                top_k=top_k,
                embeddings_cache=emb_cache,
            )
 
    del emb_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
 
    # apply all updates atomically
    new_node_texts = {ntype: list(texts) for ntype, texts in snapshot.items()}
    for ntype, new_texts in updates.items():
        new_node_texts[ntype] = new_texts
 
    return new_node_texts
 
 
def _build_longformer_cache(
    texts: list[str],
    batch_size: int = 8,
) -> dict[str, np.ndarray]:
    """
    Encode a list of unique texts with Longformer and return a
    { text: l2_normalised_embedding } cache dict.
 
    Processing in small batches prevents OOM when the neighbour pool is large.
 
    Args:
        texts      : list of unique text strings to encode
        batch_size : texts per Longformer forward pass
 
    Returns:
        dict mapping each text to its L2-normalised float32 embedding [D].
    """
    cache: dict[str, np.ndarray] = {}
    unique_texts = list(dict.fromkeys(texts))   # deduplicate, preserve order
    total = len(unique_texts)
 
    for i in range(0, total, batch_size):
        batch = unique_texts[i : i + batch_size]
        embs  = get_longformer_embedding(batch)   # Tensor [B, D] or ndarray
        if isinstance(embs, torch.Tensor):
            if embs.dim() == 1:
                embs = embs.unsqueeze(0)   # [D] → [1, D] for single-text batches
            embs = embs.cpu().numpy()
        embs = np.atleast_2d(embs)         # guard for ndarray returning 1D
        # L2-normalise each row
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-8, 1.0, norms)
        embs  = (embs / norms).astype(np.float32)
        for text, emb in zip(batch, embs):
            cache[text] = emb
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
 
    return cache
 
 
def _get_domain_dataset_map(
    model_idx: int,
    data:      "HeteroData",
    node_texts: dict[str, list[str]],
) -> dict[str, list[str]]:
    """
    For a given model node, return a mapping of
        { domain_name: [dataset_name, ...] }
    by following dataset_to_domain edges in the graph.
 
    This is used to group benchmark scores under their parent domain
    in the LLM prompt, so the model can reason about domain-level performance.
 
    Args:
        model_idx  : index of the model node
        data       : HeteroData graph
        node_texts : current { node_type: [text, ...] }
 
    Returns:
        dict mapping domain feature text → list of dataset feature texts
        that this model is connected to AND that belong to that domain.
    """
    # Step 1: collect dataset node indices this model is connected to
    model_ds_ids: set[int] = set()
    for (src_type, rel, dst_type), edge_store in data.edge_items():
        if "edge_index" not in edge_store:
            continue
        ei = edge_store.edge_index
        if src_type == "model" and dst_type == "dataset":
            mask = ei[0] == model_idx
            model_ds_ids.update(ei[1][mask].tolist())
        elif src_type == "dataset" and dst_type == "model":
            mask = ei[1] == model_idx
            model_ds_ids.update(ei[0][mask].tolist())
 
    if not model_ds_ids:
        return {}
 
    # Step 2: for each domain, find which of this model's datasets belong to it
    # by following dataset_to_domain or domain_to_dataset edges
    domain_to_ds: dict[str, list[str]] = {}
 
    for (src_type, rel, dst_type), edge_store in data.edge_items():
        if "edge_index" not in edge_store:
            continue
        ei = edge_store.edge_index
 
        if src_type == "dataset" and dst_type == "domain":
            for ds_idx, dom_idx in zip(ei[0].tolist(), ei[1].tolist()):
                if ds_idx in model_ds_ids:
                    dom_text = node_texts["domain"][dom_idx]
                    ds_text  = node_texts["dataset"][ds_idx]
                    domain_to_ds.setdefault(dom_text, []).append(ds_text)
 
        elif src_type == "domain" and dst_type == "dataset":
            for dom_idx, ds_idx in zip(ei[0].tolist(), ei[1].tolist()):
                if ds_idx in model_ds_ids:
                    dom_text = node_texts["domain"][dom_idx]
                    ds_text  = node_texts["dataset"][ds_idx]
                    domain_to_ds.setdefault(dom_text, []).append(ds_text)
 
    return domain_to_ds
 
 
def _get_dataset_embs_for_model(
    model_idx:        int,
    data:             HeteroData,
    node_texts:       dict[str, list[str]],
    embeddings_cache: dict[str, np.ndarray],
) -> list[np.ndarray]:
    """
    Return the L2-normalised embeddings of all dataset nodes directly connected
    to a given model node (via model_to_dataset edges).
 
    These embeddings serve as the anchor when ranking query neighbours: queries
    are ranked by their similarity to the dataset(s) they belong to, which
    reflects how well each query aligns with the benchmarks this model is
    evaluated on.
 
    Args:
        model_idx        : index of the model node
        data             : HeteroData graph
        node_texts       : current node feature texts
        embeddings_cache : pre-built { text: l2_normed_emb }
 
    Returns:
        List of dataset embeddings (may be empty if no dataset edges exist).
    """
    ds_embs: list[np.ndarray] = []
    for (src_type, rel, dst_type), edge_store in data.edge_items():
        if src_type != "model" or dst_type != "dataset":
            continue
        if "edge_index" not in edge_store:
            continue
        ei   = edge_store.edge_index
        mask = ei[0] == model_idx
        for ds_idx in ei[1][mask].tolist():
            ds_text = node_texts["dataset"][ds_idx]
            ds_emb  = embeddings_cache.get(ds_text)
            if ds_emb is not None:
                ds_embs.append(ds_emb)
    return ds_embs
 
 
def _direct_neighbours_of_node(
    node_idx:         int,
    node_type:        str,
    data:             HeteroData,
    node_texts:       dict[str, list[str]],
    anchor_emb:       np.ndarray | None,
    ds_anchor_emb:    np.ndarray | None,
    embeddings_cache: dict[str, np.ndarray],
    top_k:            int,
    exclude_type:     str = "model",   # never step back to the model
) -> list[dict]:
    """
    Collect the top-k direct neighbours of an arbitrary node (not just model).
 
    Used for 2-hop expansion: given a 1-hop neighbour, find ITS neighbours.
 
    Args:
        node_idx         : index of the pivot node
        node_type        : type string of the pivot node (e.g. "dataset")
        data             : HeteroData graph
        node_texts       : { ntype: [text, ...] }
        anchor_emb       : L2-normed embedding to rank candidates against
        ds_anchor_emb    : alternative anchor for query-type candidates
        embeddings_cache : pre-built { text: emb } dict
        top_k            : max candidates to keep per neighbour type
        exclude_type     : node type to exclude (avoid looping back to model)
 
    Returns:
        list of neighbour dicts (same schema as _collect_neighbours output,
        but without "children" field — these are leaf nodes).
    """
    candidates: list[dict] = []
 
    for (src_type, rel, dst_type), edge_store in data.edge_items():
        if "edge_index" not in edge_store:
            continue
        ei       = edge_store.edge_index
        has_attr = hasattr(edge_store, "edge_attr") and edge_store.edge_attr is not None
 
        # ── pivot is destination ───────────────────────────────────────────────
        if dst_type == node_type:
            mask    = ei[1] == node_idx
            src_ids = ei[0][mask].tolist()
            scores  = (edge_store.edge_attr[mask, 0].tolist()
                       if has_attr else [None] * len(src_ids))
            for sid, sc in zip(src_ids, scores):
                if src_type == exclude_type:
                    continue
                nb_text = node_texts[src_type][sid]
                nb_emb  = embeddings_cache.get(nb_text)
                anc     = ds_anchor_emb if src_type == "query" else anchor_emb
                sim     = float(np.dot(anc, nb_emb)) if (anc is not None and nb_emb is not None) else 0.0
                candidates.append({"type": src_type, "text": nb_text, "score": sc, "_sim": sim})
 
        # ── pivot is source ───────────────────────────────────────────────────
        elif src_type == node_type:
            mask    = ei[0] == node_idx
            dst_ids = ei[1][mask].tolist()
            scores  = (edge_store.edge_attr[mask, 0].tolist()
                       if has_attr else [None] * len(dst_ids))
            for did, sc in zip(dst_ids, scores):
                if dst_type == exclude_type:
                    continue
                nb_text = node_texts[dst_type][did]
                nb_emb  = embeddings_cache.get(nb_text)
                anc     = ds_anchor_emb if dst_type == "query" else anchor_emb
                sim     = float(np.dot(anc, nb_emb)) if (anc is not None and nb_emb is not None) else 0.0
                candidates.append({"type": dst_type, "text": nb_text, "score": sc, "_sim": sim})
 
    # deduplicate by text (a node may be reachable via multiple edge types)
    seen: set[str] = set()
    unique: list[dict] = []
    for c in candidates:
        if c["text"] not in seen:
            seen.add(c["text"])
            unique.append(c)
 
    if len(unique) > top_k:
        unique.sort(key=lambda x: x["_sim"], reverse=True)
        unique = unique[:top_k]
 
    for c in unique:
        c.pop("_sim", None)
 
    return unique
 
 
def _collect_neighbours(
    model_idx:        int,
    data:             HeteroData,
    node_texts:       dict[str, list[str]],
    top_k:            int = 5,
    hop_depth:        int = 1,
    embeddings_cache: dict[str, np.ndarray] | None = None,
) -> list[dict]:
    """
    Collect neighbours of a model node up to `hop_depth` hops.
 
    hop_depth=1 : return direct (1-hop) neighbours only.
                  Each dict has keys: type, text, score, domains.
                  For each neighbour type, keep at most top_k entries
                  ranked by cosine similarity to the model text.
 
    hop_depth=2 : for every selected 1-hop neighbour, additionally collect
                  ITS top-k direct neighbours (excluding the model type).
                  These 2-hop neighbours are stored in "children" on each
                  1-hop dict.  The prompt renderer displays them indented
                  beneath the parent, making the graph path explicit.
 
    Similarity anchors:
      - architecture, dataset, domain : model's own embedding
      - query                         : mean embedding of model's direct datasets
 
    Args:
        model_idx        : index of the model node
        data             : HeteroData graph
        node_texts       : current { ntype: [text, ...] } mapping
        top_k            : max neighbours per type per level
        hop_depth        : 1 or 2
        embeddings_cache : pre-built { text: l2_normed_emb }
 
    Returns:
        List of 1-hop neighbour dicts, each optionally containing "children".
    """
    if embeddings_cache is None:
        embeddings_cache = {}
 
    self_text = node_texts["model"][model_idx]
    self_emb  = embeddings_cache.get(self_text)
 
    # dataset-mean anchor for query similarity
    ds_embs = _get_dataset_embs_for_model(model_idx, data, node_texts, embeddings_cache)
    if ds_embs:
        ds_anchor = np.mean(np.stack(ds_embs, axis=0), axis=0).astype(np.float32)
        norm = np.linalg.norm(ds_anchor)
        ds_anchor = ds_anchor / norm if norm > 1e-8 else ds_anchor
    else:
        ds_anchor = self_emb
 
    # ── collect 1-hop neighbours ───────────────────────────────────────────────
    # { type_str: [ (node_idx, text, score, sim) ] }
    type_candidates: dict[str, list[tuple[int, str, float | None, float]]] = {}
 
    for (src_type, rel, dst_type), edge_store in data.edge_items():
        if "edge_index" not in edge_store:
            continue
        ei       = edge_store.edge_index
        has_attr = hasattr(edge_store, "edge_attr") and edge_store.edge_attr is not None
 
        if dst_type == "model":
            mask    = ei[1] == model_idx
            src_ids = ei[0][mask].tolist()
            scores  = (edge_store.edge_attr[mask, 0].tolist()
                       if has_attr else [None] * len(src_ids))
            for sid, sc in zip(src_ids, scores):
                nb_text = node_texts[src_type][sid]
                nb_emb  = embeddings_cache.get(nb_text)
                anc     = ds_anchor if src_type == "query" else self_emb
                sim     = float(np.dot(anc, nb_emb)) if (anc is not None and nb_emb is not None) else 0.0
                type_candidates.setdefault(src_type, []).append((sid, nb_text, sc, sim))
 
        elif src_type == "model":
            mask    = ei[0] == model_idx
            dst_ids = ei[1][mask].tolist()
            scores  = (edge_store.edge_attr[mask, 0].tolist()
                       if has_attr else [None] * len(dst_ids))
            for did, sc in zip(dst_ids, scores):
                nb_text = node_texts[dst_type][did]
                nb_emb  = embeddings_cache.get(nb_text)
                anc     = ds_anchor if dst_type == "query" else self_emb
                sim     = float(np.dot(anc, nb_emb)) if (anc is not None and nb_emb is not None) else 0.0
                type_candidates.setdefault(dst_type, []).append((did, nb_text, sc, sim))
 
    # select top_k per type, deduplicate by text within each type
    all_1hop: list[dict] = []
    for ntype, cands in type_candidates.items():
        seen: set[str] = set()
        unique = [c for c in cands if not (c[1] in seen or seen.add(c[1]))]
        if len(unique) > top_k:
            unique.sort(key=lambda x: x[3], reverse=True)
            unique = unique[:top_k]
        for (nidx, text, sc, _) in unique:
            all_1hop.append({"type": ntype, "text": text, "score": sc, "_node_idx": nidx})
 
    # ── attach domain info to dataset entries ──────────────────────────────────
    domain_dataset_map = _get_domain_dataset_map(model_idx, data, node_texts)
    ds_to_domains: dict[str, list[str]] = {}
    for dom_text, ds_texts in domain_dataset_map.items():
        for ds_text in ds_texts:
            ds_to_domains.setdefault(ds_text, []).append(dom_text)
 
    for nb in all_1hop:
        nb.pop("_sim", None)
        if nb["type"] == "dataset":
            nb["domains"] = ds_to_domains.get(nb["text"], [])
 
    # ── 2-hop expansion ────────────────────────────────────────────────────────
    if hop_depth >= 2:
        for nb in all_1hop:
            pivot_idx  = nb.pop("_node_idx", None)
            pivot_type = nb["type"]
            if pivot_idx is None:
                continue
            children = _direct_neighbours_of_node(
                node_idx=pivot_idx,
                node_type=pivot_type,
                data=data,
                node_texts=node_texts,
                anchor_emb=embeddings_cache.get(nb["text"]),
                ds_anchor_emb=ds_anchor,
                embeddings_cache=embeddings_cache,
                top_k=top_k,
                exclude_type="model",
            )
            # attach domain info to child datasets
            for ch in children:
                if ch["type"] == "dataset":
                    ch["domains"] = ds_to_domains.get(ch["text"], [])
            nb["children"] = children
    else:
        for nb in all_1hop:
            nb.pop("_node_idx", None)
 
    return all_1hop
 
# ── LLM inference via vLLM ────────────────────────────────────────────────────
 
def _run_vllm_batch(
    prompts:        list[str],
    llm:            LLM,
    sampling_params: SamplingParams,
) -> list[str]:
    """
    Run a batch of prompts through vLLM and return the generated texts.
 
    Args:
        prompts         : list of prompt strings
        llm             : initialised vLLM LLM instance
        sampling_params : vLLM SamplingParams
 
    Returns:
        list of generated output strings, same length as prompts.
    """
    outputs = llm.generate(prompts, sampling_params)
    return [out.outputs[0].text.strip() for out in outputs]
 
 
# ── Longformer encoding ────────────────────────────────────────────────────────
 
def _encode_texts(
    texts:      list[str],
    batch_size: int = 8,
) -> np.ndarray:
    """
    Encode a list of strings into Longformer embeddings via
    get_longformer_embedding(), processing in small batches to avoid OOM.
 
    Args:
        texts      : list of input strings
        batch_size : texts per forward pass (default 8; reduce if OOM)
 
    Returns:
        np.ndarray of shape [N, D], dtype float32.
    """
    all_embs: list[np.ndarray] = []
    total = len(texts)
 
    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        emb   = get_longformer_embedding(batch)
        if isinstance(emb, torch.Tensor):
            emb = emb.cpu().numpy()
        all_embs.append(emb.astype(np.float32))
 
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
 
        print(f"  Encoded {min(i + batch_size, total)}/{total} texts")
 
    return np.concatenate(all_embs, axis=0)   # [N, D]
 
 
# ── One hop of Text-GNN ───────────────────────────────────────────────────────
 
def _text_propagate_single_call(
    data:            HeteroData,
    node_texts:      dict[str, list[str]],
    model_names:     list[str],
    llm:             LLM,
    sampling_params: SamplingParams,
    top_k:           int = 5,
    hop_depth:       int = 1,
) -> dict[str, list[str]]:
    """
    Execute a single LLM aggregation pass for all model nodes.
 
    K (hop_depth) controls how many layers of the graph are visible in each
    prompt, but vLLM is called exactly ONCE regardless of K:
      - hop_depth=1 : each model sees its direct neighbours only
      - hop_depth=2 : each model sees its 1-hop neighbours AND their neighbours
                      (rendered as an indented tree in the prompt)
      - hop_depth=N : N-level subtree around each model node
 
    This is fundamentally different from iterative GNN propagation where each
    hop updates node texts before the next hop reads them.  Here the raw graph
    structure up to depth N is presented in a single prompt and the LLM
    synthesises all of it in one pass.
 
    For each model node:
      1. Collect top-k neighbours per type up to hop_depth levels.
      2. Build a structured (tree-shaped) prompt.
      3. Batch all prompts through vLLM in one call.
      4. Replace each model node's text with the LLM-generated summary.
 
    Args:
        data            : HeteroData graph (for edge structure and edge_attr)
        node_texts      : current { node_type: [text, ...] } for all node types
        model_names     : ordered list of model node name strings
        llm             : initialised vLLM LLM instance
        sampling_params : vLLM SamplingParams
        top_k           : max neighbours per type per level
        hop_depth       : depth of neighbourhood subtree shown in each prompt
 
    Returns:
        Updated node_texts dict with model texts replaced by LLM-generated
        summaries.  All other node types are unchanged.
    """
    n_models = len(model_names)
    print(f"\n  Building prompts for {n_models} model nodes (top_k={top_k}, hop_depth={hop_depth}) ...")
 
    # ── pre-build Longformer embedding cache (only texts actually needed) ───────
    # IMPORTANT: do NOT encode all node_texts indiscriminately.
    # query nodes alone can add tens of thousands of texts, which exhausts GPU
    # memory before vLLM even starts.  Instead, collect only the texts that
    # will actually be looked up during _collect_neighbours():
    #   1. every model node's own text (self anchor)
    #   2. every direct neighbour's text reachable from any model node
    # This keeps the cache small regardless of how many query nodes exist.
    print("  Collecting texts needed for top-K selection ...")
    needed_texts: set[str] = set()
 
    # model self-texts (used as anchors for arch/dataset similarity)
    for t in node_texts["model"]:
        needed_texts.add(t)
 
    # collect all texts reachable within hop_depth from any model node
    for (src_type, rel, dst_type), edge_store in data.edge_items():
        if "edge_index" not in edge_store:
            continue
        ei = edge_store.edge_index   # [2, E]
 
        if dst_type == "model":
            for src_idx in ei[0].tolist():
                needed_texts.add(node_texts[src_type][src_idx])
 
        elif src_type == "model":
            for dst_idx in ei[1].tolist():
                needed_texts.add(node_texts[dst_type][dst_idx])
 
    # for 2-hop: also include texts of nodes adjacent to 1-hop neighbours
    if hop_depth >= 2:
        one_hop_by_type: dict[str, set[int]] = {}
        for (src_type, rel, dst_type), edge_store in data.edge_items():
            if "edge_index" not in edge_store:
                continue
            ei = edge_store.edge_index
            if dst_type == "model":
                for idx in ei[0].tolist():
                    one_hop_by_type.setdefault(src_type, set()).add(idx)
            elif src_type == "model":
                for idx in ei[1].tolist():
                    one_hop_by_type.setdefault(dst_type, set()).add(idx)
        for (src_type, rel, dst_type), edge_store in data.edge_items():
            if "edge_index" not in edge_store:
                continue
            ei = edge_store.edge_index
            pivot_nbs = one_hop_by_type.get(src_type, set())
            for s, d in zip(ei[0].tolist(), ei[1].tolist()):
                if s in pivot_nbs and dst_type != "model":
                    needed_texts.add(node_texts[dst_type][d])
            pivot_nbs = one_hop_by_type.get(dst_type, set())
            for s, d in zip(ei[0].tolist(), ei[1].tolist()):
                if d in pivot_nbs and src_type != "model":
                    needed_texts.add(node_texts[src_type][s])
 
    print(f"  Encoding {len(needed_texts)} unique neighbour texts "
          f"(skipped {sum(len(v) for k, v in node_texts.items() if k != 'model') - len(needed_texts) + len(node_texts['model'])} "
          f"unreachable texts) ...")
    embeddings_cache = _build_longformer_cache(list(needed_texts), batch_size=8)
 
    # free GPU cache before handing control to vLLM
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
 
    print(f"  Longformer cache built: {len(embeddings_cache)} unique texts.")
 
    prompts: list[str] = []
    for model_idx, model_name in enumerate(model_names):
        self_text  = node_texts["model"][model_idx]
        neighbours = _collect_neighbours(model_idx, data, node_texts,
                                         top_k=top_k,
                                         hop_depth=hop_depth,
                                         embeddings_cache=embeddings_cache)
        prompt     = _build_prompt(model_name, self_text, neighbours,
                                   hop_depth=hop_depth)
        prompts.append(prompt)
 
    # release Longformer cache before vLLM inference to free GPU memory
    del embeddings_cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
 
    print(f"  Running vLLM inference (single call, hop_depth={hop_depth}) ...")
    summaries = _run_vllm_batch(prompts, llm, sampling_params)
 
    # update model node texts; all other node types unchanged
    new_node_texts = {ntype: list(texts) for ntype, texts in node_texts.items()}
    new_node_texts["model"] = summaries
 
    for i, (name, summary) in enumerate(zip(model_names, summaries)):
        preview = summary[:100].replace("\n", " ")
        print(f"    [{i}] {name}: {preview}...")
 
    return new_node_texts
 
 
# ── Main Text-GNN entry point ─────────────────────────────────────────────────
 
# Models whose embeddings and texts are saved by default.
# Set keep_names=None in text_propagate() / save_output() to save all models.
TARGET_MODELS: list[str] = [
    "qwen2.5-7b-instruct",
    "gemma-2-9b-it",
    "llama-3.1-8b-instruct",
    "mixtral-8x7b-instruct-v0.1",
    "mixtral-8x22b-instruct-v0.1",
    "llama-3.2-3b-instruct",
    "mistral-small-24b-instruct-2501-bf16",
    "llama-3.3-70b-instruct",
]
 
 
def text_propagate(
    data:                 HeteroData,
    K:                    int   = 1,
    vllm_model:           str   = DEFAULT_VLLM_MODEL,
    max_new_tokens:       int   = DEFAULT_MAX_NEW_TOKENS,
    temperature:          float = 0.1,
    tensor_parallel_size: int   = 1,
    top_k:                int   = 5,
    keep_names:           list[str] | None = TARGET_MODELS,
) -> TextGNNOutput:
    """
    Run K hops of Text-GNN aggregation using an LLM as the aggregate function.
 
    At each hop every model node gathers its top-K most text-similar neighbours
    (capped per neighbour type to control context window size), and an LLM
    fuses them into a new summary text.
 
    After all hops, BERT encodes the final summary texts into dense embeddings.
    Only models in `keep_names` are included in the returned TextGNNOutput;
    pass keep_names=None to return all models.
 
    Args:
        data                 : HeteroData graph with .node_feature_text on all
                               node types and .edge_attr on model<->dataset edges.
        K                    : number of propagation hops.
                               K=0 skips LLM entirely (Longformer-encode raw texts).
                               K>0 runs K rounds; each round updates ALL node types
                               (dataset, domain, architecture, model) using the
                               hop-start snapshot.  query nodes are never updated.
        vllm_model           : HuggingFace model ID to load via vLLM.
        max_new_tokens       : maximum tokens the LLM may generate per node.
        temperature          : LLM sampling temperature (use ~0.1 for determinism).
        tensor_parallel_size : number of GPUs for vLLM tensor parallelism.
        top_k                : max neighbours per type included in each prompt.
                               Neighbours are ranked by cosine similarity of their
                               node_feature_text to the model node's own text.
        keep_names           : model name keys to include in the output.
                               Defaults to TARGET_MODELS. Pass None to keep all.
 
    Returns:
        TextGNNOutput with model_texts, model_embeddings, and hop_texts
        (hop_texts has keys 0=original and 1=LLM output),
        filtered to keep_names if provided.
    """
    if K < 0:
        raise ValueError(f"K must be >= 0 for Text-GNN, got {K}.")
 
    model_names: list[str] = data["model"].node_names
 
    # ── initialise node texts from graph (original texts) ─────────────────────
    node_texts: dict[str, list[str]] = {
        ntype: list(data[ntype].node_feature_text)
        for ntype in data.node_types
    }
 
    hop_texts: dict[int, dict[str, str]] = {
        0: {name: node_texts["model"][i] for i, name in enumerate(model_names)}
    }
 
    # ── K=0: skip LLM entirely, encode raw node_feature_text directly ──────────
    if K == 0:
        print("\n── K=0: skipping LLM aggregation, using raw node_feature_text ──")
    else:
        # ── initialise vLLM ────────────────────────────────────────────────
        print(f"\n── Initialising vLLM  model='{vllm_model}' ────────────")
        llm = LLM(
            model=vllm_model,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_new_tokens,
        )
 
        # ── K hops: each hop updates ALL node types from a frozen snapshot ────
        # Node types updated per hop (query excluded — raw text preserved):
        #   dataset, domain, architecture, model
        # All types read from the hop-start snapshot so no type benefits from
        # another type's update within the same hop.
        UPDATE_TYPES = ["dataset", "domain", "architecture", "model"]
 
        for hop in range(1, K + 1):
            print(f"\n── Hop {hop}/{K} — updating all node types ─────────────────────────")
            node_texts = _propagate_all_nodes_one_hop(
                data=data,
                node_texts=node_texts,
                model_names=model_names,
                llm=llm,
                sampling_params=sampling_params,
                hop=hop,
                top_k=top_k,
                update_types=[t for t in UPDATE_TYPES if t in node_texts],
            )
            hop_texts[hop] = {
                name: node_texts["model"][i]
                for i, name in enumerate(model_names)
            }
 
    # ── BERT encode final model texts ─────────────────────────────────────────────
    # K=0: encodes original node_feature_text
    # K>0: encodes LLM-generated summary from the last hop
    print("\n── Encoding final model texts with Longformer ────────────")
    final_texts  = [node_texts["model"][i] for i in range(len(model_names))]
    emb_matrix   = _encode_texts(final_texts)   # [N_models, D]
 
    model_embeddings: dict[str, np.ndarray] = {
        name: emb_matrix[i]
        for i, name in enumerate(model_names)
    }
    model_texts: dict[str, str] = {
        name: node_texts["model"][i]
        for i, name in enumerate(model_names)
    }
 
    # ── filter to keep_names if requested ─────────────────────────────────────
    if keep_names is not None:
        missing = [n for n in keep_names if n not in model_texts]
        if missing:
            print(f"  Warning: requested model names not found in graph: {missing}")
        keep_set         = set(keep_names)
        model_texts      = {k: v for k, v in model_texts.items()      if k in keep_set}
        model_embeddings = {k: v for k, v in model_embeddings.items() if k in keep_set}
        hop_texts        = {
            h: {k: v for k, v in texts.items() if k in keep_set}
            for h, texts in hop_texts.items()
        }
        print(f"  Filtered output to {len(model_texts)} models: {list(model_texts.keys())}")
 
    return TextGNNOutput(
        model_texts=model_texts,
        model_embeddings=model_embeddings,
        hop_texts=hop_texts,
    )
 
 
# ── Save / load ───────────────────────────────────────────────────────────────
 
def save_output(
    output:    TextGNNOutput,
    emb_path:  str = "text_gnn_embeddings.npz",
    text_path: str = "text_gnn_texts.json",
) -> None:
    """
    Save TextGNNOutput to disk.
 
    Embeddings → .npz  (float32, { model_name: array [768] })
    Texts      → .json ({ "hop_texts": {...}, "final_texts": {...} })
 
    Args:
        output    : TextGNNOutput returned by text_propagate()
        emb_path  : output path for the .npz embedding archive
        text_path : output path for the .json text archive
    """
    import json
 
    # embeddings
    np.savez(emb_path, **output.model_embeddings)
    print(f"  Embeddings saved → '{emb_path}'")
 
    # texts (all hops + final)
    text_data = {
        "hop_texts":   {str(k): v for k, v in output.hop_texts.items()},
        "final_texts": output.model_texts,
    }
    with open(text_path, "w", encoding="utf-8") as f:
        json.dump(text_data, f, indent=2, ensure_ascii=False)
    print(f"  Texts saved      → '{text_path}'")
 
 
def load_output(
    emb_path:  str = "text_gnn_embeddings.npz",
    text_path: str = "text_gnn_texts.json",
) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """
    Load previously saved Text-GNN output.
 
    Returns:
        embeddings : { model_name: np.ndarray [768] }
        final_texts: { model_name: summary_text }
    """
    import json
 
    loaded    = np.load(emb_path)
    embeddings = {k: loaded[k] for k in loaded.files}
 
    with open(text_path, "r", encoding="utf-8") as f:
        text_data = json.load(f)
 
    return embeddings, text_data["final_texts"]
 
 
# ── Summary helper ────────────────────────────────────────────────────────────
 
def print_output_summary(output: TextGNNOutput) -> None:
    """Print a human-readable summary of TextGNNOutput."""
    max_hop = max(output.hop_texts.keys())
    K = max_hop   # for display; always 0 (K=0 baseline) or 1 (single LLM call)
    print(f"\n=== Text-GNN Output Summary  (K={K}) ===")
    print(f"  Models : {len(output.model_texts)}")
    print(f"  Emb dim: {next(iter(output.model_embeddings.values())).shape[0]}")
 
    print("\n--- Per-hop text lengths (chars) ---")
    header = f"  {'model':<35}" + "".join(f" hop{h:>2}" for h in range(K + 1))
    print(header)
    print("  " + "-" * (35 + 7 * (K + 1)))
    for name in output.model_texts:
        row = f"  {name:<35}"
        for h in range(K + 1):
            row += f" {len(output.hop_texts[h][name]):>6}"
        print(row)
 
    print("\n--- Final summaries (preview) ---")
    for name, text in output.model_texts.items():
        print(f"  [{name}]")
        print(f"    {text[:120].replace(chr(10), ' ')}...")
 
 
# ── Main ──────────────────────────────────────────────────────────────────────
 
def main(
    graph_path:           str   = "llm_hetero_graph.pt",
    K:                    int   = 1,
    vllm_model:           str   = DEFAULT_VLLM_MODEL,
    max_new_tokens:       int   = DEFAULT_MAX_NEW_TOKENS,
    temperature:          float = 0.1,
    tensor_parallel_size: int   = 1,
    emb_save_path:        str   = "text_gnn_embeddings.npz",
    text_save_path:       str   = "text_gnn_texts.json",
) -> None:
    """
    End-to-end Text-GNN pipeline:
      load graph → run K-hop LLM aggregation → BERT encode → save outputs.
 
    When K=0, no LLM is initialised; the raw node_feature_text of each model
    node is encoded directly by BERT and returned as the embedding baseline.
    """
    # 1. Load graph
    print(f"── Step 1: Load graph from '{graph_path}' ────────────────")
    data = torch.load(graph_path, weights_only=False)
    print(data)
 
    # 2. Run Text-GNN
    print(f"\n── Step 2: Text-GNN  K={K}  model='{vllm_model}' ───────")
    output = text_propagate(
        data=data,
        K=K,
        vllm_model=vllm_model,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        tensor_parallel_size=tensor_parallel_size,
    )
 
    # 3. Summary
    print_output_summary(output)
 
    # 4. Save
    print(f"\n── Step 3: Save outputs ──────────────────────────────────")
    save_output(output, emb_save_path, text_save_path)
 
    print("\n✅ Done!")

if __name__ == "__main__":
    import argparse
    import os

    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    _PD = os.path.join(ROOT_DIR, "profile_data")
    _PR = os.path.join(ROOT_DIR, "routeprofile")

    parser = argparse.ArgumentParser(description="Text-GNN with LLM aggregation.")
    parser.add_argument("--mode",        choices=["standard", "newllm"], default="standard",
                        help="Routing setting: standard or newllm (default: standard)")
    parser.add_argument("--graph",       default=None,
                        help="Input .pt graph file "
                             "(default: profile_data/result_data_graph/{mode}/query_task_domain_graph_full.pt)")
    parser.add_argument("--K",           default=4, type=int,
                        help="Number of Text-GNN hops (0 = BERT-only baseline, no LLM)")
    parser.add_argument("--model",       default=DEFAULT_VLLM_MODEL,
                        help="vLLM model ID")
    parser.add_argument("--max-tokens",  default=DEFAULT_MAX_NEW_TOKENS, type=int,
                        help="Max new tokens per LLM call")
    parser.add_argument("--temperature", default=0.0, type=float,
                        help="LLM sampling temperature")
    parser.add_argument("--tp",          default=1, type=int,
                        help="Tensor parallel size (number of GPUs)")
    parser.add_argument("--emb-save",    default=None,
                        help="Output path for embeddings "
                             "(default: routeprofile/model_profile_result/{mode}/text_gnn.npz)")
    parser.add_argument("--text-save",   default=None,
                        help="Output path for texts "
                             "(default: routeprofile/model_profile_result/{mode}/text_gnn_texts.json)")
    args = parser.parse_args()

    _save_dir = os.path.join(_PR, "model_profile_result", args.mode)
    os.makedirs(_save_dir, exist_ok=True)

    main(
        graph_path=args.graph or os.path.join(_PD, "result_data_graph", args.mode, "query_task_domain_graph_full.pt"),
        K=args.K,
        vllm_model=args.model,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        tensor_parallel_size=args.tp,
        emb_save_path=args.emb_save or os.path.join(_save_dir, "text_gnn.npz"),
        text_save_path=args.text_save or os.path.join(_save_dir, "text_gnn_texts.json"),
    )