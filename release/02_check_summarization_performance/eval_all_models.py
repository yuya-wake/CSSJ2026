"""
This script evaluates the summarization performance of multiple models on the XL-Sum dataset (Japanese).
"""
import gc
import os
import re
import time

import pandas as pd
import torch
import evaluate
from datasets import load_dataset
from llama_cpp import Llama
from summac.model_summac import SummaCZS

print("import OK")

MODEL_PATHS = [
    os.path.join("path", "to", "models", "gpt-oss-20b-Q4_K_M.gguf"),
    os.path.join("path", "to", "models", "calm3-22b-chat-Q4_K_M.gguf"),
    os.path.join("path", "to", "models", "qwen2-7b-instruct-q5_k_m.gguf"),
    os.path.join("path", "to", "models", "c4ai-command-r-plus-Q4_K_M.gguf"),
]

OUTPUT_CSV_PATHS = [
    os.path.join("results", "evaluation_summary_gpt_oss_20b_q4_k_m.csv"),
    os.path.join("results", "evaluation_summary_calm3_22b_chat_q4_k_m.csv"),
    os.path.join("results", "evaluation_summary_qwen2_7b_instruct_q5_k_m.csv"),
    os.path.join("results", "evaluation_summary_c4ai_command_r_plus_q4_k_m.csv"),
]

RESULTS_DIR = os.path.join("results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# llama.cpp generation/runtime parameters
N_CTX = 16384
N_BATCH = 512
MAX_TOKENS = 5000
TEMPERATURE = 0.0
STOP_TOKENS = ["###"]  # NOTE: Not used by llama_cpp chat API unless you pass stop=...
NUM_SAMPLES_TO_TEST = 889

# Prompting: force Japanese summary only (no analysis, no tags, no English).
SYSTEM_PROMPT = "日本語で要約のみを出力。解説・思考・英語・タグは出力しない。"
USER_TEMPLATE = "次の記事を日本語で要約してください。\n\n{article}"


# ======================================
# Dataset loading
# ======================================
# XL-Sum Japanese test split
dataset = load_dataset("csebuetnlp/xlsum", "japanese", split="test")
num_samples = min(NUM_SAMPLES_TO_TEST, len(dataset))
dataset_subset = dataset.select(range(num_samples))


# ======================================
# Post-processing helpers
# ======================================
# Some chat templates may contain special markers like:
#   <|channel|>final<|message|> ...
# This function tries to extract the final answer safely.
FINAL_MARK = "<|channel|>final<|message|>"


def extract_final(text: str) -> str:
    """Extract the final answer part from a model output that may include template markers."""
    if not text:
        return ""

    # Prefer the last "final" section if present.
    idx = text.rfind(FINAL_MARK)
    if idx != -1:
        s = text[idx + len(FINAL_MARK):]
    else:
        # Fallback: take the content after the last <|message|> marker, if any.
        parts = re.split(r"<\|message\|>", text)
        s = parts[-1] if len(parts) >= 2 else text

    # Truncate if another template tag starts (e.g., next turn).
    s = re.split(r"<\|start\|>|<\|end\|>|<\|channel\|>", s)[0]

    # Remove any remaining template tags.
    s = re.sub(r"<\|.*?\|>", "", s)

    return s.strip()


# ======================================
# Metrics (initialized once)
# ======================================
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

# SummaCZS initialization is heavy; do it once.
summac = SummaCZS(granularity="sentence", model_name="vitc")


def summac_score_one(source: str, summary: str) -> float:
    """
    Compute SummaCZS for one (source, summary) pair.

    SummaCZS.score() argument names differ across versions, so we support both:
      - score(sources=[...], generateds=[...])
      - score([sources], [generateds])  # positional
    """
    try:
        out = summac.score(sources=[source], generateds=[summary])
    except TypeError:
        out = summac.score([source], [summary])

    # Handle multiple possible output formats (dict/list/float).
    if isinstance(out, dict):
        if "scores" in out:
            return float(out["scores"][0])
        if "score" in out:
            return float(out["score"])
    if isinstance(out, list):
        return float(out[0])
    return float(out)


# ======================================
# Main loop: evaluate each local GGUF model
# ======================================
for model_path, output_csv_path in zip(MODEL_PATHS, OUTPUT_CSV_PATHS):
    llm = None
    generated_summaries: list[str] = []
    references: list[str] = []
    sources: list[str] = []
    total_gen_time_sec = 0.0

    try:
        # Load model once per evaluation run.
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,  # offload as many layers as possible (requires GPU support)
            n_ctx=N_CTX,
            n_batch=N_BATCH,
            low_cpu_mem_usage=True,
            verbose=False,
        )

        for i in range(num_samples):
            source_text = dataset_subset[i]["text"]
            reference_summary = dataset_subset[i]["summary"]

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(article=source_text)},
            ]

            t0 = time.perf_counter()
            resp = llm.create_chat_completion(
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                # If you want to enforce stop tokens:
                # stop=STOP_TOKENS,
            )
            t1 = time.perf_counter()
            total_gen_time_sec += (t1 - t0)

            raw_text = resp["choices"][0]["message"]["content"]
            pred_summary = extract_final(raw_text)

            generated_summaries.append(pred_summary)
            references.append(reference_summary)
            sources.append(source_text)

            if (i + 1) % 50 == 0:
                print(f"[{os.path.basename(model_path)}] {i+1}/{num_samples} summaries generated")

    finally:
        # Always free resources even if an error occurs mid-run.
        if llm is not None:
            try:
                llm.close()
            except Exception:
                pass

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------
    # Metric computation (per model)
    # --------------------------------------
    rouge_res = rouge.compute(
        predictions=generated_summaries,
        references=references,
        use_stemmer=True,
    )

    bs_res = bertscore.compute(
        predictions=generated_summaries,
        references=references,
        lang="ja",
        model_type="tohoku-nlp/bert-base-japanese-whole-word-masking",
        num_layers=8,
    )
    bs_p = sum(bs_res["precision"]) / len(bs_res["precision"])
    bs_r = sum(bs_res["recall"]) / len(bs_res["recall"])
    bs_f1 = sum(bs_res["f1"]) / len(bs_res["f1"])

    # SummaCZS: average across all samples (can be slow).
    summac_scores = [summac_score_one(src, gen) for src, gen in zip(sources, generated_summaries)]
    summac_avg = sum(summac_scores) / len(summac_scores)

    # --------------------------------------
    # Save one-row CSV
    # --------------------------------------
    row = {
        "Model": os.path.basename(model_path),
        "Num Samples": num_samples,
        "ROUGE-1": rouge_res["rouge1"],
        "ROUGE-2": rouge_res["rouge2"],
        "ROUGE-L": rouge_res["rougeL"],
        "BERTScore Precision": bs_p,
        "BERTScore Recall": bs_r,
        "BERTScore F1": bs_f1,
        "SummaCZS Avg": summac_avg,
        "Total Generation Time (s)": total_gen_time_sec,
        "Avg Time per Summary (s)": (total_gen_time_sec / num_samples) if num_samples else float("nan"),
    }

    pd.DataFrame([row]).to_csv(output_csv_path, index=False, float_format="%.6f")
    print(f"Saved: {output_csv_path}")

print("DONE")