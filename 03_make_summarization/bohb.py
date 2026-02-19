"""
BOHB-style hyperparameter search (Optuna) for recursive summarization quality.

What this script does
---------------------
- Samples (model_name, compression_rate) with Optuna.
- For each transcript, generates a recursive summary (RAPTOR-like iterative chunk summarization).
- Scores factual consistency of (original -> summary) using SummaC-ZS.
- Aggregates scores into a single objective value and maximizes it.

Design choices / notes
----------------------
- This script assumes ALL candidate LMs are *GGUF* models executed via llama-cpp-python.
  (i.e., `Llama.create_chat_completion(...)` is used for generation.)
- A small JSON cache is used to avoid recomputing intermediate/final summaries.
- Japanese sentence splitting uses spaCy's lightweight sentencizer.

Before publishing
-----------------
- Replace MODEL_PATHS with your own local paths OR load them from environment variables.
- Confirm your GGUF models are chat-tuned and support the "system/user" roles.
- Ensure dependencies are documented (llama-cpp-python, optuna, summac, spacy, etc.).
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
import sys
import time
import json
import gc
import io
import random
import hashlib
import tempfile
import logging
from contextlib import contextmanager
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import spacy
import torch
from llama_cpp import Llama, llama_cpp as _lc

import optuna
from summac.model_summac import SummaCZS

# ==============================================================================
# Reproducibility / logging
# ==============================================================================
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)

optuna.logging.set_verbosity(optuna.logging.INFO)

@contextmanager
def suppress_stderr():
    """
    Context manager to temporarily suppress stderr.
    Useful for noisy libraries (e.g., llama-cpp).
    """
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stderr.close()
        sys.stderr = original_stderr

# ==============================================================================
# Model paths (EDIT ME)
# ==============================================================================
MODEL_PATHS = {
    # NOTE: Replace these placeholders with real paths on your machine.
    "calm3": "/path/to/local_models/calm3-22b-chat-Q4_K_M.gguf",
    "gpt-oss": "/path/to/local_models/gpt-oss-20b-Q4_K_M.gguf",
    "qwen2": "/path/to/local_models/qwen2-7b-instruct-q5_k_m.gguf"
}

# ==============================================================================
# Japanese sentence splitter (lightweight)
# ==============================================================================
# Using a blank Japanese pipeline + sentencizer keeps overhead low.
nlp = spacy.blank("ja")
sentencizer = nlp.add_pipe("sentencizer")
sentencizer.punct_chars = ["。", "！", "？", "!", "?", "｡", "．", "."]

# ==============================================================================
# Japanese sentence splitter (lightweight)
# ==============================================================================
print("Loading SummaC model...")
summac_model = SummaCZS(model_name="vitc", gran="sentence", device="cpu")
print("SummaC model loaded.")

# ==============================================================================
# Generation / experiment settings
# ==============================================================================
MAX_CHARS = 1536                     # per chunk (character-based)
MAX_TOKENS_PER_CHUNK = 1024          # max tokens for intermediate summaries
N_BATCH = 512                        # llama.cpp batch size
N_CTX = 2048                         # llama.cpp context window
TEMPERATURE = 0.0                    # deterministic by default
DO_SAMPLE = False                    # deterministic by default

NUM_DF = 100                         # number of documents used in the study
N_TRIAL = 50                         # number of Optuna trials


# ==============================================================================
# Utility: model loading and scoring
# ==============================================================================
def load_model_and_tokenizer(model_name):
    """
    Load a GGUF model via llama-cpp-python.

    Parameters
    ----------
    model_name : str
        One of keys in MODEL_PATHS.

    Returns
    -------
    Llama
        A llama-cpp model instance.
    """
    print(f"\nLoading model: {model_name}...")
    with suppress_stderr():
        os.environ.setdefault(
            "LD_LIBRARY_PATH",
            os.environ.get("LD_LIBRARY_PATH", "")
        )

        cuda_ok = _lc.llama_supports_gpu_offload()
        print(f"[llama-cpp] supports_gpu_offload = {cuda_ok}")

        llm = Llama(
            model_path=MODEL_PATHS[model_name],
            n_gpu_layers=-1,
            n_ctx=N_CTX,
            n_batch=N_BATCH,
            verbose=False
        )
    print(f"{model_name} loaded.")

    return llm, None

def evaluate_summary_with_summac(original, generated):
    """
    Evaluate factual consistency of a summary against the original using SummaC-ZS.

    Returns
    -------
    float
        SummaC score in [0, 1] (typically).
    """
    if not generated or not original:
        return 0.0

    score = summac_model.score([original], [generated])
    return score["scores"][0]

# ==============================================================================
# RAPTOR-like splitting and chunking
# ==============================================================================
def split_into_sentences(text, nlp_model):
    """Split Japanese text into sentences using spaCy sentencizer."""
    doc = nlp_model(text)
    return [sent.text.strip() for sent in doc.sents]

def aggregate_sentences_to_chunks(sentences, max_chars):
    """
    Aggregate sentences into chunks by character length.
    This is a simple heuristic; token-based chunking is possible but heavier.
    """
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# ==============================================================================
# Optuna objective aggregation
# ==============================================================================
def aggregate_summac(scores):
    """
    Aggregate per-document SummaC scores into a single objective value.
    - mean: main target
    - p10: robustness (penalize worst-ish cases)
    - std: stability penalty
    """
    s = np.array(scores, dtype=float)
    mean = float(np.mean(s))
    p10  = float(np.percentile(s, 10))
    std  = float(np.std(s))
    return mean + 0.25 * p10 - 0.10 * std

# ==============================================================================
# Summary caching (to avoid recomputation)
# ==============================================================================
CACHE_DIR = os.path.expanduser("~/.cache/raptor_summ_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_VERSION = "v1"

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def _cache_key(prefix: str, payload: str) -> str:
    h = _sha1(payload)
    return os.path.join(CACHE_DIR, f"{prefix}_{CACHE_VERSION}_{h}.json")

def _atomic_write_json(path: str, obj: dict):
    # 競合・途中破損を避けるための原子書き込み
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", dir=CACHE_DIR)
    try:
        with io.open(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
        os.replace(tmp, path)  # atomic
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass

def load_from_cache(path: str):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f).get("summary")
        except Exception:
            return None
    return None

def save_to_cache(path: str, summary: str):
    _atomic_write_json(path, {"summary": summary})

# ==============================================================================
# Prompting and generation
# ==============================================================================
def build_prompt(text: str, purpose: str | int = "recursive") -> Tuple[str, str]:
    """
    Create (system_prompt, user_prompt).

    purpose:
    - "recursive": intermediate summarization step
    - int: final summary target length in characters
    """
    system = (
        "あなたは医療分野の編集者です．"
        "事実の追加・訂正・反論・注意喚起・脚色は禁止．"
        "中間要約に現れない内容や評価を加えない．"
        "出力は本文のみ．挨拶・謝辞・依頼文・長さ報告・参考文献・署名・リンク・HTML/Markdown・絵文字は禁止．"
    )

    if purpose == "recursive":
        user_instr = "以下の中間要約だけを根拠として，自然で流暢な要約を作成してください．"
    else:
        target_char = int(purpose)
        user_instr = f"以下の文章だけを根拠として，約{target_char}文字の自然で流暢な要約を作成してください．"

    user = (
        f"{user_instr}\n"
        "・【制約】非加筆・非推測・非訂正・脚色禁止・非注意喚起禁止\n"
        "・主観/価値判断/評価語を加えない（例：正しい/誤っている/危険 などの断定は禁止）\n"
        "・語尾は平叙。呼びかけ・勧誘・まとめ語（例：以上）は使わない\n"
        "・曖昧な表現は避け，根拠は原文の内容に限定\n\n"
        f"【中間要約】\n{text}\n\n"
        "【出力】："
    )

    return system, user

def generate_summary_llamacpp(
    llm: Llama,
    model_name: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> str:
    """
    Generate a summary using llama.cpp chat completion.

    Notes
    -----
    - Assumes the GGUF model supports chat roles (system/user).
    - For deterministic outputs, use temperature=0.0 and do_sample=False behavior.
    """
    gen_cfg = {
        "model": model_name,
        "system": system,
        "user": user,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
    }

    payload = json.dumps(gen_cfg, ensure_ascii=False, sort_keys=True)
    cache_path = _cache_key("gen", payload)
    cached = load_from_cache(cache_path)
    if cached is not None:
        return cached

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    # llama-cpp-python returns OpenAI-like dict
    with torch.no_grad():
        resp = llm.create_chat_completion(
            messages=messages,
            max_tokens=int(max_tokens),
            temperature=float(temperature),
        )

    text = resp["choices"][0]["message"]["content"].strip()
    save_to_cache(cache_path, text)
    return text


def make_final_summary(
    original_text: str,
    intermediate_summary: str,
    rate: float,
    model_name: str,
    llm: Llama,
) -> str:
    """
    Convert the final intermediate summary into a target-length final summary.
    """
    target_char = int(len(original_text) * float(rate))
    system, user = build_prompt(intermediate_summary, purpose=target_char)

    # Rough heuristic: Japanese chars -> tokens mapping can vary heavily by model.
    # We keep generous bounds and rely on the model to stop naturally.
    max_tokens = max(64, int(target_char * 1.1 / 1.2))

    payload = json.dumps(
        {"kind": "final", "model": model_name, "rate": float(rate), "intermediate": intermediate_summary},
        ensure_ascii=False,
        sort_keys=True,
    )
    cache_path = _cache_key("final", payload)
    cached = load_from_cache(cache_path)
    if cached is not None:
        return cached

    summary = generate_summary_llamacpp(
        llm=llm,
        model_name=model_name,
        system=system,
        user=user,
        max_tokens=max_tokens,
        temperature=0.0,
    )
    save_to_cache(cache_path, summary)
    return summary


def recursive_summarize(
    text: str,
    rate: float,
    model_name: str,
    llm: Llama,
) -> str:
    """
    Recursive summarization loop:
    - Split into sentences
    - Chunk sentences into <= MAX_CHARS
    - Summarize each chunk -> concatenate
    - Repeat until a single chunk remains
    - Optionally produce a final summary to meet target compression rate
    """
    current_text = text.strip()
    if not current_text:
        return ""

    intermediate_summary = ""

    while True:
        sentences = split_into_sentences(current_text, nlp)
        if not sentences:
            return ""

        chunks = aggregate_sentences_to_chunks(sentences, MAX_CHARS)

        # Stop condition: already a single chunk (no need for another recursive step)
        if len(chunks) <= 1:
            intermediate_summary = chunks[0] if chunks else ""
            break

        # Summarize each chunk
        new_summaries: List[str] = []
        for chunk in chunks:
            system, user = build_prompt(chunk, purpose="recursive")
            summ = generate_summary_llamacpp(
                llm=llm,
                model_name=model_name,
                system=system,
                user=user,
                max_tokens=MAX_TOKENS_PER_CHUNK,
                temperature=TEMPERATURE,
            )
            new_summaries.append(summ)

        current_text = " ".join(new_summaries)

    target_length = int(len(text) * float(rate))

    # If already short enough, skip the final call (saves time/cost).
    if len(intermediate_summary) <= target_length * 1.2:
        return intermediate_summary

    return make_final_summary(
        original_text=text,
        intermediate_summary=intermediate_summary,
        rate=rate,
        model_name=model_name,
        llm=llm,
    )

# ==============================================================================
# Optuna objective
# ==============================================================================
def objective(trial: optuna.Trial, df_subset: pd.DataFrame) -> float:
    """
    Optuna objective: maximize aggregated SummaC across sampled documents.
    """
    model_name = trial.suggest_categorical("model_name", ["calm3", "gpt-oss", "qwen2"])
    compression_rate = float(trial.suggest_float("compression_rate", 0.1, 0.4, log=False))

    pbar = tqdm(
        total=len(df_subset),
        desc=f"Trial {trial.number} | {model_name} r={compression_rate:.2f}",
        ncols=100,
    )

    llm: Optional[Llama] = None
    all_scores: List[float] = []
    error_infos: List[dict] = []

    try:
        llm = load_model(model_name)

        for _, row in df_subset.iterrows():
            video_id = row.get("video_id", "N/A")
            original_text = str(row.get("transcript") or "")

            # Skip extremely short transcripts (often noise for summarization evaluation)
            if len(original_text) < 500:
                pbar.update(1)
                continue

            try:
                summary = recursive_summarize(
                    text=original_text,
                    rate=compression_rate,
                    model_name=model_name,
                    llm=llm,
                )
                score = evaluate_summary_with_summac(original_text, summary)
                all_scores.append(score)

            except Exception as e:
                # Do not fail the entire trial due to one bad sample.
                error_infos.append({"video_id": video_id, "reason": str(e)})
                all_scores.append(0.0)

            # Periodic cleanup (mostly CPU RAM); GPU VRAM is managed by llama.cpp, not torch.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            pbar.update(1)

        trial.set_user_attr("error_infos", error_infos)

        if not all_scores:
            return 0.0

        return aggregate_summac(all_scores)

    finally:
        pbar.close()
        # Ensure the llama.cpp model is released
        if llm is not None:
            del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ==============================================================================
# Main
# ==============================================================================
def main() -> None:
    start = time.monotonic()

    study_name = "raptor-bohb"
    storage_name = f"sqlite:///{study_name}.db"

    # Data path
    mnt_path = os.getenv("MNT_PATH") or ""
    csv_path = os.path.join(mnt_path, "complete_data", "real_and_fake_lt98pct.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, usecols=["video_id", "transcript"])
    df = df.drop_duplicates(subset="video_id")
    df_for_study = df.dropna(subset=["transcript"]).sample(n=NUM_DF, random_state=SEED)

    # Load or create Optuna study
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        logging.info("Found existing study '%s'. Resuming optimization.", study_name)
    except KeyError:
        logging.info("No existing study '%s'. Creating a new one.", study_name)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=SEED),
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=1,
                max_resource=len(df_for_study),
                reduction_factor=3,
            ),
        )

    # Enqueue baseline trials (one per model) so you always get at least one run for each.
    for params in [{"model_name": "calm3"}, {"model_name": "gpt-oss"}, {"model_name": "qwen2"}]:
        study.enqueue_trial(params)

    study.optimize(lambda t: objective(t, df_for_study), n_trials=N_TRIAL)

    # Print results
    logging.info("Number of finished trials: %d", len(study.trials))
    best = study.best_trial
    logging.info("Best trial value: %s", best.value)
    logging.info("Best trial params: %s", best.params)

    # Save CSV outputs
    out_dir = os.path.join(mnt_path, "results")
    os.makedirs(out_dir, exist_ok=True)

    best_csv = os.path.join(out_dir, "best_model_and_rate.csv")
    pd.DataFrame([{
        "score": best.value,
        "model_name": best.params.get("model_name"),
        "compression_rate": best.params.get("compression_rate"),
    }]).to_csv(best_csv, index=False, encoding="utf-8-sig")

    all_csv = os.path.join(out_dir, "all_trials.csv")
    df_trials = study.trials_dataframe()
    df_trials["duration_seconds"] = (
        df_trials["datetime_complete"] - df_trials["datetime_start"]
    ).dt.total_seconds()
    df_trials.to_csv(all_csv, index=False, encoding="utf-8-sig")

    # Save Optuna visualizations as HTML (requires plotly)
    try:
        import optuna.visualization  # type: ignore

        fig_history = optuna.visualization.plot_optimization_history(study)
        fig_history.write_html(os.path.join(out_dir, "optimization_history.html"))

        fig_importance = optuna.visualization.plot_param_importances(study)
        fig_importance.write_html(os.path.join(out_dir, "param_importances.html"))

        logging.info("Optuna plots saved to %s", out_dir)
    except Exception as e:
        logging.warning("Failed to generate Optuna plots (plotly missing?): %s", e)

    # Collect and save per-trial errors
    all_error_infos: List[dict] = []
    for t in study.trials:
        errs = t.user_attrs.get("error_infos", [])
        for item in errs:
            if isinstance(item, dict):
                item = dict(item)
                item["trial_number"] = t.number
                all_error_infos.append(item)

    if all_error_infos:
        err_csv = os.path.join(out_dir, "error_log.csv")
        pd.DataFrame(all_error_infos).to_csv(err_csv, index=False, encoding="utf-8-sig")

    end = time.monotonic()
    dur = end - start
    h, rem = divmod(dur, 3600)
    m, s = divmod(rem, 60)
    logging.info("Total Experiment Time: %02dh %02dm %.2fs", int(h), int(m), s)


if __name__ == "__main__":
    main()