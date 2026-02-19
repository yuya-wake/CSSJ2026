## About this repository

This repository contains the code used in the following paper presented at the **5th Annual Conference of the Computational Social Science Society of Japan**:

- 和氣 祐弥，津川 翔，天笠 俊之，「日本語医療動画のLLM真偽判定における文字起こしの入力設計の比較検討」
  *(The paper title is in Japanese and is intentionally kept as-is.)*

This repository also includes some **preliminary experiments** that were omitted from the paper for brevity.

All final results are recorded as **CSV files**. However, files that include **YouTube video transcripts** are **not published** due to the terms of the **YouTube Researcher Program**.

---

## RAPTOR code

Our RAPTOR implementation is primarily based on the original RAPTOR repository:

- https://github.com/parthsarthi03/raptor

We made several modifications for Japanese/Hugging Face compatibility. For details, please see:

- `04_evaluation/raptor/raptor_ja/README.md`

---

## Prompts

Under the `prompt/` directory, you will find two prompt files:

- `prompt_definition.txt`: an English translation of the labeling definitions used in the annotation process.
- `prompt_strict.txt`: a stricter version of the *real* criteria designed to reduce false negatives.

All results reported in the paper were obtained using **`prompt_strict.txt`**.

---

## Experimental environments

The required environment differs by folder. Please use the following conda environment files:

- `01_cleaning` to `03_make_summarization`
  - `env/gguf_environment.yml`

- `04_evaluation`
  - For scripts using **GGUF** models: `gguf_environment.yml`
  - For scripts using **Hugging Face** models (e.g., `cyberagent-calm3-22b-chat`): `hf_environment.yml`

- `05_linguistic_features`
  - `linguistic_environment.yml`
