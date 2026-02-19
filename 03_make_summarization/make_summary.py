import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
os.environ["HF_HUB_OFFLINE"] = "1"            # HF Hubへ行かない
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import gc
import numpy as np
import json
import random
import time
import sys
import pandas as pd
from tqdm import tqdm
from contextlib import contextmanager
from llama_cpp import Llama
import torch

# ==============================================================================
# 設定パラメータ (Best Params)
# ==============================================================================
random.seed(42)
torch.manual_seed(42)

BEST_MODEL_NAME = "qwen2"
BEST_COMPRESSION_RATE = 0.20633943473449157
mnt_path = os.getenv('MNT_PATH') or ''
input_csv_path = os.path.join(mnt_path, 'complete_data', 'real_and_fake_scripts.csv')
output_csv_path = os.path.join(mnt_path, 'complete_data', 'real_and_fake_w_summary.csv')
MODEL_PATH = "/home/wake/local_models/qwen2-7b-instruct-q5_k_m.gguf"


MAX_CHARS = 3000
MAX_TOKENS_PER_CHUNK = 2048
FINAL_TARGET_CHARS = 10000
N_CTX = 8192
N_BATCH = 512
TEMPERATURE = 0.0
BATCH_SIZE = 8
EVAL_MAX_CHARS = 10000

nlp = None
summac_model = None


@contextmanager
def suppress_stderr():
    """
    withブロック内の標準エラー出力を一時的に抑制するコンテキストマネージャ
    """
    original_stderr = sys.stderr
    # /dev/nullは、書き込まれたデータをすべて捨てる特殊なファイルです
    sys.stderr = open(os.devnull, 'w')
    try:
        yield
    finally:
        sys.stderr.close()
        sys.stderr = original_stderr

# ==============================================================================
# モデル・ツール読み込み
# ==============================================================================
print(f"Loading Model: {BEST_MODEL_NAME}...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

with suppress_stderr():
    os.environ.setdefault(
        "LD_LIBRARY_PATH",
        os.environ.get("LD_LIBRARY_PATH", "")
    )

    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,
        n_ctx=N_CTX,
        n_batch=N_BATCH,
        verbose=False
    )

print("Model loaded successfully.")

def ensure_nlp_and_summac():
    global nlp, summac_model
    if nlp is None:
        import spacy
        nlp = spacy.blank("ja")
        s = nlp.add_pipe("sentencizer")
        s.punct_chars = ["。","！","？","!","?","｡","．","."]
    if summac_model is None:
        from summac.model_summac import SummaCZS
        print("Loading SummaC model...")
        summac_model = SummaCZS(model_name="vitc", gran="sentence", device="cuda")
        print("SummaC model loaded.")

def evaluate_summary_with_summac(original, generated):
    """SummaCを使って要約の事実性を評価する"""
    if not generated or not original:
        return 0.0 # 要約が空の場合はスコア0

    if len(original) > EVAL_MAX_CHARS:
        original_for_eval = original[:EVAL_MAX_CHARS]
    else:
        original_for_eval = original

    try:
        # SummaCはリスト形式で入力を受け取る
        score = summac_model.score([original_for_eval], [generated])
        return score["scores"][0]
    except Exception as e:
        return 0.0

# ==============================================================================
# RAPTOR関連の関数群 (split, aggregate)
# ==============================================================================
def split_into_sentences(text, nlp_model):
    doc = nlp_model(text)
    return [sent.text.strip() for sent in doc.sents]

def aggregate_sentences_to_chunks(sentences, max_chars):
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(sentence) > max_chars:
            # これまでの塊があれば先に登録
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            # 長い文をmax_charsごとにスライスして登録
            for i in range(0, len(sentence), max_chars):
                sub_chunk = sentence[i:i+max_chars]
                # 最後の切れ端だけは次のcurrent_chunkにする（文脈を繋ぐため）
                if i + max_chars >= len(sentence):
                    current_chunk = sub_chunk # 次のループでの結合用に保持
                    # もしこの切れ端だけでもmax_charsに近いなら、ここで切ってしまっても良い
                else:
                    chunks.append(sub_chunk)
            continue

        if len(current_chunk) + len(sentence) <= max_chars:
            current_chunk += sentence
        else:
            chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# ==============================================================================
# SummaCバッチ推論
# ==============================================================================
def summac_batch_scores(pairs, batch_size):
    # pairs: list of (original, summary)
    docs  = []
    sums  = []
    out   = []
    for i, (d, s) in enumerate(pairs, 1):
        docs.append(d); sums.append(s)
        if len(docs) == batch_size or i == len(pairs):
            with torch.no_grad():
                res = summac_model.score(docs, sums, batch_size=len(docs))
            out.extend(res["scores"])
            docs.clear(); sums.clear()
    return out  # list[float]

# ==============================================================================
# プロンプト生成とLLMによる要約生成関数
# ==============================================================================
def common_prompt(text, purpose="recursive"):
    system = (
        "あなたは医療分野の編集者です．"
        "事実の追加・訂正・反論・注意喚起・脚色は禁止．"
        "中間要約に現れない内容や評価を加えない．"
        "出力は本文のみ．挨拶・謝辞・依頼文・長さ報告・参考文献・署名・リンク・HTML/Markdown・絵文字は禁止．"
    )

    if purpose == "recursive":
        user_instr = "以下の中間要約だけを根拠として，自然で流暢な要約を作成してください．"
    else: # final
        target_char = purpose # int
        user_instr = f"以下の文章だけを根拠として，約{target_char}文字の自然で流暢な要約を作成してください．"

    user = (
        f"{user_instr}\n"
        "・【制約】非加筆・非推測・非訂正・脚色禁止・非注意喚起禁止\n"
        "・主観/価値判断/評価語を加えない（例：正しい/誤っている/危険 などの断定は禁止）\n"
        "・語尾は平叙。呼びかけ・勧誘・まとめ語（例：以上）は使わない\n"
        "・曖昧な表現は避け，根拠は原文の内容に限定\n\n"
        f"【中間要約】\n{text}\n\n"
        f"【出力】："
    )
    return system, user

def summarize_with_llm(text, target_chars, is_final=False):
    """LLMで要約を生成する共通関数"""
    system_p, user_p = common_prompt(text, purpose=target_chars if is_final else "recursive")

    # トークン数見積もり
    estimated_tokens = int(target_chars * 1.5) if is_final else int(target_chars * 1.5)
    # 上限キャップ
    max_gen_tokens = min(estimated_tokens, MAX_TOKENS_PER_CHUNK)

    messages = [
        {"role": "system", "content": system_p},
        {"role": "user", "content": user_p}
    ]

    try:
        response = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_gen_tokens,
            temperature=TEMPERATURE,
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"  [LLM Error]: {e}")
        return text # エラー時は入力をそのまま返す（処理を止めないため）

def recursive_summarize(text, rate):
    # 1. 既に十分短い場合は何もしない
    if len(text) <= FINAL_TARGET_CHARS:
        return text

    current_text = text
    is_concatenated = False

    while True:
        sentences = split_into_sentences(current_text, nlp)
        if not sentences: return current_text

        chunks = aggregate_sentences_to_chunks(sentences, MAX_CHARS)

        # 終了条件A: チャンクが1つになったら終了
        if len(chunks) <= 1:
            intermediate_summary = chunks[0] if chunks else ""
            break

        # 各チャンクを要約
        summaries = []
        for chunk in chunks:
            # 中間圧縮：rateに基づいてターゲット長を計算
            chunk_target = int(len(chunk) * rate)
            chunk_target = max(chunk_target, 200) # 下限ガード

            summary = summarize_with_llm(chunk, chunk_target, is_final=False)
            summaries.append(summary)

        current_text = " ".join(summaries)
        is_concatenated = True

        # 終了条件B: 目標文字数を下回ったらループを抜ける
        if len(current_text) <= FINAL_TARGET_CHARS:
            intermediate_summary = current_text
            break

    # === 最終仕上げ ===
    current_len = len(intermediate_summary)

    # まだ長い、または継ぎ接ぎの場合は整形する
    if is_concatenated or current_len > FINAL_TARGET_CHARS:
        # ターゲットは最大でも10000文字、現在より短ければ現在の長さを維持
        target_chars = min(current_len, FINAL_TARGET_CHARS)
        final_summary = summarize_with_llm(intermediate_summary, target_chars, is_final=True)
    else:
        final_summary = intermediate_summary

    return final_summary

# ==============================================================================
# メイン実行ブロック
# ==============================================================================
def main():
    df = pd.read_csv(
        input_csv_path,
        dtype=str,
        escapechar='\\',
        doublequote=False,
        engine="python"
        )

    df['is_long'] = df['is_long'].map(lambda x: str(x).lower() == 'true')
    df['summary'] = df['transcript'].fillna('')
    target_indices = df[df['is_long'] == True].index

    ensure_nlp_and_summac()
    summac_scores = []

    pbar = tqdm(target_indices, desc="Summarizing")

    for idx in pbar:
        video_id = df.at[idx, 'video_id']
        original_text = str(df.at[idx, 'transcript'])

        pbar.set_postfix({"id": video_id})

        try:
            summary = recursive_summarize(original_text, BEST_COMPRESSION_RATE)

            # 結果をDataFrameに格納
            df.at[idx, 'summary'] = summary

            # メモリクリーンアップ
            gc.collect()

        except Exception as e:
            print(f"\nError processing {video_id}: {e}")
            # エラー時は元のテキストのまま（何もしない）

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        is_long = row['is_long']
        original = str(row['transcript'])
        summary = str(row['summary'])

        # ★★★ 要約しなかった場合（短文）は NaN にする ★★★
        if not is_long:
            summac_scores.append(np.nan)
            continue

        # ★★★ 要約した場合のみスコア計算 ★★★
        # summaryが空文字でないかチェック
        if not summary:
            # 要約すべきなのに空の場合はエラーとして0点
            summac_scores.append(0.0)
        else:
            score = evaluate_summary_with_summac(original, summary)
            summac_scores.append(score)

    # 結果をDataFrameに追加
    df['summac_score'] = summac_scores

    # 結果の保存
    print(f"Saving results to {output_csv_path}...")
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

    # 要約したデータだけの平均スコアを表示
    valid_scores = [s for s in summac_scores if not np.isnan(s)]
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        print(f"Average SummaC Score (Summarized items only): {avg_score:.4f}")
    else:
        print("No summarized items found.")

if __name__ == "__main__":
    main()