<picture>
  <source media="(prefers-color-scheme: dark)" srcset="raptor_dark.png">
  <img alt="RAPTOR banner" src="raptor.jpg">
</picture>

## RAPTOR (Japanese HF Edition): Recursive Abstractive Processing for Tree-Organized Retrieval

This repository is a **Japanese + local Hugging Face (HF) model** adaptation of **RAPTOR** (Recursive Abstractive Processing for Tree-Organized Retrieval).
Compared to the original implementation (which typically assumes OpenAI APIs and English tooling), this fork is designed to:

- Run **fully locally** with **HF chat models** (e.g., `cyberagent/calm3-22b-chat`)
- Support **Japanese preprocessing** (sentence splitting + token counting)
- Use **Japanese embedding models** for clustering (SBERT mean-tokens style)
- Improve **clustering robustness** when data is small or ill-conditioned
- Offer **VRAM-aware execution** by loading/unloading models on demand

For the original methodology and motivation, see the RAPTOR paper:

- [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)

[![Paper page](https://huggingface.co/datasets/huggingface/badges/resolve/main/paper-page-sm.svg)](https://huggingface.co/papers/2401.18059)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/raptor-recursive-abstractive-processing-for/question-answering-on-quality)](https://paperswithcode.com/sota/question-answering-on-quality?p=raptor-recursive-abstractive-processing-for)

---

## What’s new in this fork

### 1) Japanese-friendly text splitting
- Uses **GiNZA (`ja_ginza`) via spaCy**.
- Prevents Sudachi tokenizer failures for very long inputs by pre-splitting large texts into safe chunks before `nlp(...)`.
- Token budgeting is computed using the **HF tokenizer** you provide (e.g., calm3 tokenizer), not `tiktoken`.

### 2) Local HF summarization model
- Adds `LocalHFSummarizationModel` that summarizes via a **local HF causal LM** (chat format via `apply_chat_template`).
- Uses **4-bit quantization** (bitsandbytes) and exposes `close()` to explicitly free VRAM.

### 3) Local HF constrained classifier (real/fake)
- Adds `LocalHFClassificationQAModel` that runs a local chat model and **forces output to exactly one of `{real, fake}`** using:
  - A custom `LogitsProcessor` (prefix-constrained decoding)
  - A `StoppingCriteria` that stops once a valid label is generated
- Loads a strict classification prompt from `raptor/prompt/prompt_strict.txt`.

### 4) Japanese SBERT-style embedding for clustering
- Adds `JapaneseSBERTMeanTokensEmbeddingModel` (e.g., `sonoisa/sentence-bert-base-ja-mean-tokens-v2`).
- Uses mean pooling over token embeddings and optional L2 normalization.

### 5) More robust clustering (UMAP + GMM(BIC))
`cluster_utils.py` now handles edge cases safely:
- If there are too few points to cluster (`n < 3`) → **use 1 cluster**
- If GMM fitting fails for some `n_components` → **skip that candidate**
- If all candidates fail → **fallback to 1 cluster**
- If GMM fitting fails entirely (ill-defined covariance) → **assign everyone to cluster 0**

### 6) Config behavior changes
- `TreeBuilderConfig` now **requires**:
  - an explicit `summarization_model`
  - explicit `embedding_models` (non-empty dict)
  - explicit `cluster_embedding_model` key
- `RetrievalAugmentation` supports:
  - default retrieval mode via config: `tr_retrieval_mode` (`collapsed` or `traversal`)
  - optional CSV logging of generation inputs (`log_csv_path`)

### 7) End-to-end Japanese run script
- Adds `run_japanese.py` as a full pipeline example:
  1) build RAPTOR tree using cluster embeddings
  2) attach retrieval embeddings
  3) retrieve context (collapsed or traversal)
  4) classify strictly into `real` / `fake`
- Includes a `ModelManager` that **keeps only one large model on GPU at a time** (VRAM-friendly).

---

## Installation

### Requirements
- Python 3.8+
- CUDA-enabled GPU recommended (for calm3-22b-chat)
- Packages you will likely need:
  - `transformers`, `accelerate`, `bitsandbytes`
  - `torch`
  - `spacy`, `ginza`
  - `scikit-learn`, `umap-learn`, `numpy`, `pandas`, `tqdm`