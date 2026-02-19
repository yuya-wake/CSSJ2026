import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import csv
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, matthews_corrcoef

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    LogitsProcessor,
    LogitsProcessorList,
    StoppingCriteria,
    StoppingCriteriaList,
)

MNT_PATH = os.getenv('MNT_PATH') or ''
INPUT_CSV_PATH = os.path.join(MNT_PATH, 'complete_data', 'real_and_fake_w_summary.csv')

os.makedirs('/results/calm3_hf', exist_ok=True)
OUTPUT_RESULT_PATH = 'results/calm3_hf/summary_results.csv'
OUTPUT_METRICS_PATH = 'results/calm3_hf/summary_metrics_summary_calm3_hf.csv'

MODEL_PATHS = {"calm3": "/path/to/local_models/cyberagent-calm3-22b-chat"}

MODEL_CONTEXT_LIMITS = 16384
MAX_NEW_TOKENS = 5

# ======================================================================
# 定義プロンプト
# ======================================================================
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

TARGET_TEXTS = ["real", "fake"]
# token化の揺れ（先頭空白/改行）にも強くする
TARGET_VARIANTS = ["real", "fake", " real", " fake", "\nreal", "\nfake"]


# ======================================================================
# real/fake のみを許す LogitsProcessor（文法制約の代替）
# ======================================================================
class PrefixConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, prompt_len: int, allowed_token_seqs: list[list[int]], eos_token_id: int):
        super().__init__()
        self.prompt_len = prompt_len
        self.allowed = allowed_token_seqs
        self.eos = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 1サンプル想定
        gen = input_ids[0, self.prompt_len:].tolist()

        allowed_next = set()
        for seq in self.allowed:
            if len(gen) <= len(seq) and gen == seq[:len(gen)]:
                if len(gen) == len(seq):
                    allowed_next.add(self.eos)  # 完了したらEOSだけ許す
                else:
                    allowed_next.add(seq[len(gen)])

        # もし一致する候補が無い場合：保険でEOSを許す（無限ループ防止）
        if not allowed_next:
            allowed_next = {self.eos}

        mask = torch.full_like(scores, float("-inf"))
        mask[0, list(allowed_next)] = 0.0
        return scores + mask


class StopOnAnyTarget(StoppingCriteria):
    def __init__(self, prompt_len: int, tokenizer, target_texts_norm: set[str]):
        super().__init__()
        self.prompt_len = prompt_len
        self.tokenizer = tokenizer
        self.target_texts_norm = target_texts_norm

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        gen_ids = input_ids[0, self.prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip().lower()
        return text in self.target_texts_norm


# ======================================================================
# プロンプト作成 & トークン数ベースで transcript を切り詰め
# ======================================================================
def create_messages_and_truncate(row, tokenizer, context_limit: int, max_new_tokens: int):
    title = str(row.get('title') or "")
    description = str(row.get('description') or "")
    transcript = str(row.get('summary') or "")

    user_prefix = f"【タイトル】\n{title}\n\n【概要】\n{description}\n\n【文字起こし】\n"
    user_suffix = "\n\n【判定】"

    # transcript無しで組んだ時の入力長（=固定長）を測る
    base_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prefix + "" + user_suffix},
    ]
    base_ids = tokenizer.apply_chat_template(
        base_messages, add_generation_prompt=True, return_tensors="pt"
    )
    reserved_len = base_ids.shape[1] + max_new_tokens  # 生成ぶんも確保

    allowed_transcript_len = context_limit - reserved_len

    trans_ids = tokenizer.encode(transcript, add_special_tokens=False)
    original_token_len = len(trans_ids)

    is_truncated = False
    if allowed_transcript_len <= 0:
        truncated_transcript = ""
        is_truncated = True
    else:
        if len(trans_ids) > allowed_transcript_len:
            trans_ids = trans_ids[:allowed_transcript_len]
            truncated_transcript = tokenizer.decode(trans_ids, skip_special_tokens=True)
            is_truncated = True
        else:
            truncated_transcript = transcript

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prefix + truncated_transcript + user_suffix},
    ]

    truncation_info = {
        "is_truncated": is_truncated,
        "original_len": original_token_len,
        "allowed_len": max(0, allowed_transcript_len),
    }
    return messages, truncation_info


# ======================================================================
# 推論（real/fake 制約つき）
# ======================================================================
@torch.inference_mode()
def predict_real_fake(model, tokenizer, messages):
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    prompt_len = input_ids.shape[1]
    eos_id = tokenizer.eos_token_id

    # どの variant にも一致しうる token列を候補にする
    allowed_token_seqs = []
    for s in TARGET_VARIANTS:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            allowed_token_seqs.append(ids)

    logits_processor = LogitsProcessorList([
        PrefixConstrainedLogitsProcessor(prompt_len, allowed_token_seqs, eos_id)
    ])
    stopping = StoppingCriteriaList([
        StopOnAnyTarget(prompt_len, tokenizer, {t.strip().lower() for t in TARGET_VARIANTS})
    ])

    out = model.generate(
        input_ids=input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        temperature=None,
        top_p=None,
        logits_processor=logits_processor,
        stopping_criteria=stopping,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
    )

    gen_ids = out[0, prompt_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip().lower()

    # 最終的に "real" or "fake" に正規化
    if "real" == text or text.endswith("real"):
        return "real"
    if "fake" == text or text.endswith("fake"):
        return "fake"
    return "Error"


# ======================================================================
# メイン
# ======================================================================
def main():
    print(f"Loading data from {INPUT_CSV_PATH}")
    df = pd.read_csv(INPUT_CSV_PATH, dtype=str, encoding="utf-8")

    df = df[df['is_long'] == 'True']

    results = []
    summary_metrics = []

    for model_name, model_path in MODEL_PATHS.items():
        print(f"\n{'='*30}\nEvaluating Model: {model_name}\n{'='*30}")

        # ---- tokenizer/model load ----
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

        use_4bit = True
        if use_4bit:
            qconfig = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                device_map="auto",
                quantization_config=qconfig,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                device_map="auto",
                torch_dtype="auto",
                attn_implementation="sdpa",
            )

        model.eval()

        truncated_log_path = f'results/calm3_hf/truncated_log_{model_name}_hf.csv'
        os.makedirs(os.path.dirname(truncated_log_path), exist_ok=True)
        with open(truncated_log_path, 'w', encoding='utf-8') as f:
            f.write("video_id,original_tokens,allowed_tokens,overflow_tokens\n")

        y_true, y_pred = [], []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            video_id = row['video_id']
            true_label = str(row['label']).strip().lower()  # "real"/"fake"想定

            messages, trunc_info = create_messages_and_truncate(
                row, tokenizer, MODEL_CONTEXT_LIMITS, MAX_NEW_TOKENS
            )

            if trunc_info["is_truncated"]:
                original = trunc_info["original_len"]
                allowed = trunc_info["allowed_len"]
                overflow = original - allowed
                with open(truncated_log_path, 'a', encoding='utf-8') as f:
                    f.write(f"{video_id},{original},{allowed},{overflow}\n")

            try:
                prediction = predict_real_fake(model, tokenizer, messages)
            except Exception as e:
                print(f"Error processing {video_id}: {e}")
                prediction = "Error"

            results.append({
                "model": model_name,
                "video_id": video_id,
                "true_label": true_label,
                "prediction": prediction,
                "is_correct": (prediction == true_label),
            })

            if prediction in ["real", "fake"]:
                y_true.append(true_label)
                y_pred.append(prediction)

        # --- metrics ---
        if y_true:
            labels = ["real", "fake"]
            acc = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, labels=labels, average=None, zero_division=0
            )
            _, _, f1_weighted, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
            cm = confusion_matrix(y_true, y_pred, labels=labels)
            mcc = matthews_corrcoef(y_true, y_pred)

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
                "cm_act_fake_pred_fake": cm[1][1],
                "mcc": mcc,
            })
        else:
            print("No valid predictions made.")

        # 片付け
        del model
        torch.cuda.empty_cache()

    # 保存
    result_df = pd.DataFrame(results)
    print(f"\nSaving results to {OUTPUT_RESULT_PATH}")
    result_df.to_csv(OUTPUT_RESULT_PATH, index=False, encoding='utf-8-sig')

    metrics_df = pd.DataFrame(summary_metrics)
    print(f"Saving summary metrics to {OUTPUT_METRICS_PATH}")
    metrics_df.to_csv(OUTPUT_METRICS_PATH, index=False, encoding='utf-8-sig')

    print("Done!")


if __name__ == "__main__":
    main()