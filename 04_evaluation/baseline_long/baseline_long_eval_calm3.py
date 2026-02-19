import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import csv
import pandas as pd
import torch
from tqdm import tqdm
from llama_cpp import Llama, LlamaGrammar
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

MNT_PATH = os.getenv('MNT_PATH') or ''
INPUT_CSV_PATH = os.path.join(MNT_PATH, 'complete_data', 'real_and_fake_w_summary.csv')

os.makedirs('results/calm3', exist_ok=True)
OUTPUT_RESULT_PATH = 'results/calm3/baseline_long_results.csv'
OUTPUT_METRICS_PATH = 'results/calm3/baseline_long_metrics_summary_qwen2.csv'

MODEL_PATHS = {"calm3": "/path/to/local_models/calm3-22b-chat-Q4_K_M.gguf"}

MODEL_CONTEXT_LIMITS = 16384
# MODEL_CONTEXT_LIMITS = {
#     "qwen2": 32768,
#     "calm3": 16384,
#     "gpt-oss": 32768
# }

N_GPU_LAYERS = -1
N_BATCH = 512

# ==============================================================================
# 定義プロンプト (User provided)
# ==============================================================================
SYSTEM_PROMPT = """You are a medical fact-checker.
Classify the following video content (Title, Description, Transcript) as 'Real' or 'Fake' based on the detailed definitions below.

[DEFINITIONS]
real:
The content of the post is entirely factual and provides accurate information in its main points. Minor factual errors or ambiguous expressions that do not affect the core message are also included in this category. Additionally, the following cases are considered "real":
a) The post is based on accurate facts but includes some exaggeration that may lead to misunderstandings.
b) The risks or benefits are overly emphasized, potentially giving a different impression from the facts. However, if believing the exaggeration would have an extremely low likelihood of harming the viewers' health, it is still classified as "real."

fake:
The content of the post is entirely false, or it draws incorrect conclusions from accurate facts. It also includes posts that deliberately emphasize baseless information to mislead the audience. Additionally, the following cases are considered "fake":
a) The post is based on accurate facts but includes some exaggeration that may lead to misunderstandings.
b) The risks or benefits are overly emphasized, potentially giving a different impression from the facts, and if believing the exaggeration poses a potential risk to the viewers' health, it is classified as "fake."

[Classification Policy]
Videos will be classified as "real" if they pose an extremely low risk of causing health-related harm to viewers.
They will be classified as "fake" if there is a potential risk of causing health-related harm.
Also, classify content that, even if it is a personal opinion not based on specific data or literature, has the potential to mislead viewers and instill false perceptions as Fake, rather than Opinion.
The Opinion category should be limited to videos that neither provide accurate information nor cause viewers to acquire incorrect information. For example, text that solely describes an individual's personal experience with a certain illness would fall under the Opinion category.

Output only one word: 'real' or 'fake'.
"""

# ==============================================================================
# "Real" or "Fake" を強制する文法定義 (GBNF)
# ==============================================================================
# これにより、モデルは "Real" か "Fake" しか出力できなくなります
GBNF_GRAMMAR = r'root ::= ("real" | "fake")'

def get_grammar():
    return LlamaGrammar.from_string(GBNF_GRAMMAR)

# ==============================================================================
# プロンプト作成と切り捨て処理
# ==============================================================================
def create_prompt_and_truncate(row, model, context_limit):
    """
    コンテキスト長に収まるようにtranscriptを切り詰めてプロンプトを作成する
    """
    title = str(row['title'] or "")
    description = str(row['description'] or "")
    transcript = str(row['transcript'] or "")

    # 1. 固定部分（システムプロンプト + 枠組み）
    user_header = f"\n【タイトル】\n{title}\n\n【概要】\n{description}\n\n【文字起こし】\n"
    user_footer = "\n\n【判定】"

    # 2. 固定部分のトークン数を計算
    sys_tokens = model.tokenize(SYSTEM_PROMPT.encode("utf-8"), add_bos=True)
    header_tokens = model.tokenize(user_header.encode("utf-8"), add_bos=False)
    footer_tokens = model.tokenize(user_footer.encode("utf-8"), add_bos=False)

    # 予約トークン数（システム + ヘッダー + フッター + 出力用数トークン）
    reserved_len = len(sys_tokens) + len(header_tokens) + len(footer_tokens) + 20

    # 3. transcriptに使える残りのトークン数
    allowed_transcript_len = context_limit - reserved_len

    trans_tokens = model.tokenize(transcript.encode("utf-8"), add_bos=False)
    original_token_len = len(trans_tokens)

    is_truncated = False

    if allowed_transcript_len <= 0:
        truncated_transcript = ""
        print(f"Warning: Metadata too long for video {row.get('video_id')}")
        is_truncated = True
    else:
        # transcriptをトークン化して切り詰める
        trans_tokens = model.tokenize(transcript.encode("utf-8"), add_bos=False)
        if len(trans_tokens) > allowed_transcript_len:
            # 切り捨て
            trans_tokens = trans_tokens[:allowed_transcript_len]
            # デコードして文字列に戻す（不完全なバイト列エラーを防ぐため ignore）
            truncated_transcript = model.detokenize(trans_tokens).decode("utf-8", errors="ignore")
            is_truncated = True
        else:
            truncated_transcript = transcript

    # 4. 最終プロンプトの組み立て
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"【タイトル】\n{title}\n\n【概要】\n{description}\n\n【文字起こし】\n{truncated_transcript}\n\n【判定】"}
    ]

    truncation_info = {
        "is_truncated": is_truncated,
        "original_len": original_token_len,         # 元の長さ
        "allowed_len": max(0, allowed_transcript_len) # 許容された長さ
    }

    return messages, truncation_info

# ==============================================================================
# メイン処理
# ==============================================================================
def main():
    # データの読み込み
    print(f"Loading data from {INPUT_CSV_PATH}")
    df = pd.read_csv(
        INPUT_CSV_PATH,
        dtype=str,
        encoding="utf-8",
        sep=",",
        quotechar='"',
        doublequote=True,
    )

    df = df[df['is_long'] == 'True']

    # 結果保存用
    results = []
    summary_metrics = []

    # 文法オブジェクトの作成
    grammar = get_grammar()

    for model_name, model_path in MODEL_PATHS.items():
        print(f"\n{'='*30}")
        print(f"Evaluating Model: {model_name}")
        print(f"{'='*30}")

        # モデルのロード
        n_ctx = MODEL_CONTEXT_LIMITS
        try:
            llm = Llama(
                model_path=model_path,
                n_gpu_layers=N_GPU_LAYERS,
                n_ctx=n_ctx,
                n_batch=N_BATCH,
                verbose=False
            )
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue

        truncated_log_path = f'results/calm3/truncated_log_{model_name}.csv'
        with open(truncated_log_path, 'w', encoding='utf-8') as f:
            f.write("video_id,original_tokens,allowed_tokens,overflow_tokens\n")

        y_true = []
        y_pred = []

        for idx, row in tqdm(df.iterrows(), total=len(df)):
            video_id = row['video_id']
            true_label = row['label'] # "Real" or "Fake" (CSVの形式に合わせる)

            # プロンプト作成（切り詰め処理付き）
            messages, trunc_info = create_prompt_and_truncate(row, llm, n_ctx)

            if trunc_info["is_truncated"]:
                try:
                    original = trunc_info["original_len"]
                    allowed = trunc_info["allowed_len"]
                    overflow = original - allowed

                    with open(truncated_log_path, 'a', encoding='utf-8') as f:
                        f.write(f"{video_id},{original},{allowed},{overflow}\n")
                except Exception as e:
                    print(f"Failed to write log: {e}")

            try:
                # 推論実行
                response = llm.create_chat_completion(
                    messages=messages,
                    max_tokens=10, # "Real"か"Fake"だけなので短くていい
                    temperature=0.0, # 決定的にする
                    grammar=grammar # ここで出力を強制する
                )

                prediction = response['choices'][0]['message']['content'].strip()

            except Exception as e:
                print(f"Error processing {video_id}: {e}")
                prediction = "Error"

            # 結果の記録
            is_correct = (prediction.lower() == str(true_label).lower())
            results.append({
                "model": model_name,
                "video_id": video_id,
                "true_label": true_label,
                "prediction": prediction,
                "is_correct": (prediction == true_label)
            })

            if prediction in ["real", "fake"]:
                y_true.append(true_label)
                y_pred.append(prediction)


        # メモリ解放
        del llm
        torch.cuda.empty_cache()

# --- 指標計算と表示 ---
        if y_true:
            # 正解ラベルの順序を固定して混同行列を見やすくする
            labels = ["real", "fake"]

            acc = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=labels, average=None, zero_division=0
            )
            # weighted average (全体のF1など)
            _, _, f1_weighted, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )

            cm = confusion_matrix(y_true, y_pred, labels=labels)

            summary_metrics.append({
                "model": model_name,
                "accuracy": acc,
                "f1_weighted": f1_weighted,
                "real_precision": precision[0],
                "real_recall": recall[0],
                "real_f1": f1[0],
                "fake_precision": precision[1],
                "fake_recall": recall[1],
                "fake_f1": f1[1],
                "cm_act_real_pred_real": cm[0][0],
                "cm_act_real_pred_fake": cm[0][1],
                "cm_act_fake_pred_real": cm[1][0],
                "cm_act_fake_pred_fake": cm[1][1]
            })
        else:
            print("No valid predictions made.")

    # 結果の保存
    result_df = pd.DataFrame(results)
    print(f"\nSaving results to {OUTPUT_RESULT_PATH}")
    result_df.to_csv(OUTPUT_RESULT_PATH, index=False, encoding='utf-8-sig')

    metrics_df = pd.DataFrame(summary_metrics)
    print(f"Saving summary metrics to {OUTPUT_METRICS_PATH}")
    metrics_df.to_csv(OUTPUT_METRICS_PATH, index=False, encoding='utf-8-sig')

    print("Done!")

if __name__ == "__main__":
    main()