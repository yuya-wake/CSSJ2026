import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
import pandas as pd
from typing import List
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

# ==========================================
# 設定
# ==========================================
MODEL_PATH = "/home/wake/local_models/cyberagent-calm3-22b-chat"
MNT_PATH = os.getenv('MNT_PATH') or ''
INPUT_CSV_PATH = os.path.join(MNT_PATH, 'complete_data', 'real_and_fake_w_summary.csv')
STEP1_JSON_PATH = os.path.join(MNT_PATH, 'screening', 'screening_results_long.json')
OUTPUT_CSV_PATH = "final_classification_metrics_long.csv"
OUTPUT_DETAILS_PATH = os.path.join(MNT_PATH, "screening", "final_classification_details_long.csv")

MAX_NEW_TOKENS = 5

# SYSTEM PROMPT (Strict Version)
SYSTEM_PROMPT = """You are a rigorous medical fact-checker.
Classify the following text as 'Real' or 'Fake' based on strict medical evidence standards.

[CORE DEFINITIONS]
real:
The content is scientifically accurate and consistent with standard medical consensus.
- Personal experiences are presented clearly as subjective opinions, not as general medical facts.
- Health advice aligns with public health guidelines (e.g., "eat vegetables," "sleep well").

fake:
The content includes scientific inaccuracies, unproven medical claims, or exaggerations that could mislead viewers.
**Any content falling under the following categories must be classified as 'fake', regardless of direct harm risk:**

1. **"Miracle Cures" & Exaggerated Efficacy**:
   - Claims that specific foods, supplements, or rituals can "cure" or "prevent" serious diseases (Cancer, Stroke, Diabetes, etc.) without clinical evidence.
   - Usage of strong terms like "Miracle," "Instant cure," "100% effective," or "Vanished" regarding diseases.
   - Example: "This oil reduces cancer rate by 62%," "Breakfast that cures constipation instantly."

2. **Pseudoscience & Spiritualism in Medicine**:
   - Associating medical conditions solely with spiritual energy, "Qi," ancestors, or vibrations.
   - Promoting non-medical remedies (e.g., Flower Remedy) as substitutes for medical treatment.

3. **Fear-Mongering & Conspiracy**:
   - Claims that standard medical treatments (medicines, vaccines, surgery) are "poison" or "evil."
   - Extreme exaggeration of risks associated with common foods or additives (e.g., "Eating this causes immediate stroke").

[DECISION RULE]
- If the text makes a medical claim (e.g., "A causes B" or "A cures B") that is not supported by standard medical literature, classify it as **'fake'**.
- Do NOT give the benefit of the doubt. If it sounds like a "too good to be true" health hack, it is likely **'fake'**.
- **Safety is not the only criteria.** Even if following the advice is physically harmless (e.g., drinking water, eating specific food), spreading false medical causality is considered 'fake'.

Output only one word: 'real' or 'fake'.
"""

# ======================================================================
# real/fake のみを許す LogitsProcessor（文法制約の代替）
# ======================================================================
TARGET_TEXTS = ["real", "fake"]
# token化の揺れ（先頭空白/改行）にも強くする
TARGET_VARIANTS = ["real", "fake", " real", " fake", "\nreal", "\nfake"]

class PrefixConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, prompt_len: int, allowed_token_seqs: list[list[int]], eos_token_id: int):
        super().__init__()
        self.prompt_len = prompt_len
        self.allowed = allowed_token_seqs
        self.eos = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # 1サンプル想定（あなたのコードは逐次処理なのでOK）
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
# プロンプト作成
# ======================================================================
def create_messages(context_input):
    user_prefix = f"【対象テキスト（抜粋または全文）】\n"
    user_suffix = "\n\n【判定】"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prefix + context_input + user_suffix},
    ]
    return messages

# ======================================================================
# 重複文章削除
# ======================================================================
def merge_text_fragments(fragments: List[str], min_overlap: int = 10) -> str:
    """
    文字列のリストを受け取り、隣接する要素間の重複（オーバーラップ）を除去して結合する。

    Args:
        fragments (List[str]): 結合したいテキストのリスト
        min_overlap (int): 重なりとみなす最小の文字数（誤結合防止のため）

    Returns:
        str: 統合されたテキスト
    """
    if not fragments:
        return ""

    # 1. 完全一致の重複を排除（順序は維持）
    unique_fragments = []
    seen = set()
    for text in fragments:
        if text not in seen:
            unique_fragments.append(text)
            seen.add(text)

    if not unique_fragments:
        return ""

    merged_text = unique_fragments[0]

    # 2. 前後のオーバーラップをチェックして結合
    for i in range(1, len(unique_fragments)):
        current = unique_fragments[i]
        previous = merged_text

        # ケースA: currentがpreviousに完全に含まれている場合はスキップ
        if current in previous:
            continue

        # ケースB: オーバーラップを探す
        # previousの末尾とcurrentの先頭が一致する最大長を探す
        overlap_found = False
        check_len = min(len(previous), len(current))

        # 長い一致から順に試す
        for length in range(check_len, min_overlap - 1, -1):
            # previousの末尾length文字 == currentの先頭length文字
            if previous.endswith(current[:length]):
                # 重なり部分を除いて結合 (previous + currentの残り)
                merged_text += current[length:]
                overlap_found = True
                break

        # ケースC: 重なりがない場合は改行でつなぐ
        if not overlap_found:
            merged_text += "\n……\n" + current # わかりやすく区切り線を入れても良い

    return merged_text

# ==========================================
# Step 3 判定
# ==========================================
@torch.inference_mode()
def predict_real_fake(model, tokenizer, messages):
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)

    prompt_len = input_ids.shape[1]
    eos_id = tokenizer.eos_token_id

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
        do_sample=False,          # 決定的
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
    print("Loading Data...")
    with open(STEP1_JSON_PATH, "r", encoding="utf-8") as f:
        step1_results = json.load(f)

    df_csv = pd.read_csv(
        INPUT_CSV_PATH,
        dtype=str,
        encoding="utf-8",
        sep=",",
        quotechar='"',
        doublequote=True,
    )

    csv_data_map = {}
    for _, row in df_csv.iterrows():
        vid = row.get('video_id')
        csv_data_map[vid] = {
            'transcript': row.get('transcript'),
            'label': row.get('label')
        }

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        device_map="auto",
        quantization_config=qconfig,
        dtype=torch.bfloat16,
    ).eval()

    final_predictions = []
    y_true = []
    y_pred = []
    summary_metrics = []

    print("Starting Step 3 Verification...")
    for item in tqdm(step1_results):
        video_id = item.get('video_id')
        flagged_count = item.get('flagged_count', 0)

        csv_info = csv_data_map.get(video_id)
        ground_truth = csv_info['label']
        full_transcript = csv_info['transcript']

        prediction = ""
        logic_type = ""

        # パターンA: flagged_count が 0 -> Real (自動判定)
        if flagged_count == 0:
            prediction = "real"
            logic_type = "auto_real_step1"
        # パターンB: flagged_count > 0 -> Step 3 (LLM判定)
        else:
            candidates = item.get('candidates', [])

            # requires_full_context_check: False のものがあるか探す
            valid_contexts = [
                c.get('context_step2', '')
                for c in candidates
                if not c.get('requires_full_context_check', False)
            ]

            context_input = ""

            if valid_contexts:
                # 信頼できるStep2コンテキストがある場合 -> 連結して入力
                context_input = merge_text_fragments(valid_contexts)
                logic_type = "llm_partial_context"
            else:
                # すべて失敗/チェック要の場合 -> 全文入力
                context_input = full_transcript
                logic_type = "llm_full_transcript"

            # コンテキストが空の場合（エラー等）のガード
            if not context_input.strip():
                context_input = full_transcript
                logic_type = "llm_full_transcript_fallback"

            # LLM推論実行
            messages = create_messages(context_input)
            prediction = predict_real_fake(model, tokenizer, messages)

        if prediction in ["real", "fake"]:
            final_predictions.append({
                "video_id": video_id,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "logic_type": logic_type,
                "step1_flagged_count": flagged_count
            })

            y_true.append(ground_truth)
            y_pred.append(prediction)

    # 3. 評価指標の計算と保存
    df_pred = pd.DataFrame(final_predictions)
    df_pred.to_csv(OUTPUT_DETAILS_PATH, index=False, encoding="utf-8")

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

        print(f"Accuracy:  {acc:.4f}")
        print(f"Weighted F1: {f1_weighted:.4f}")
        print("\n[Class-wise Metrics]")
        print(f"Real -> Prec: {precision[0]:.4f}, Rec: {recall[0]:.4f}, F1: {f1[0]:.4f}")
        print(f"Fake -> Prec: {precision[1]:.4f}, Rec: {recall[1]:.4f}, F1: {f1[1]:.4f}")
        print("\n[Confusion Matrix]")
        print("        Pred:Real  Pred:Fake")
        print(f"Act:Real   {cm[0][0]:<10} {cm[0][1]}")
        print(f"Act:Fake   {cm[1][0]:<10} {cm[1][1]}")

        print("\n[Classification Report]")
        print(classification_report(
            y_true, y_pred, labels=labels, target_names=labels, zero_division=0
        ))
        summary_metrics.append({
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

    del model
    torch.cuda.empty_cache()
    metrics_df = pd.DataFrame(summary_metrics)
    metrics_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8')

    print('Done!')

if __name__ == "__main__":
    main()