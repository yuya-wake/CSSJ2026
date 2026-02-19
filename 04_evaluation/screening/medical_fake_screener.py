import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import re
import json
import pandas as pd
from typing import List, Dict
from tqdm import tqdm
import difflib

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# ==========================================
# 設定
# ==========================================
MODEL_PATH = "/home/wake/local_models/cyberagent-calm3-22b-chat"
MNT_PATH = os.getenv('MNT_PATH') or ''
INPUT_CSV_PATH = os.path.join(MNT_PATH, 'complete_data', 'real_and_fake_w_summary.csv')
OUTPUT_JSON_PATH = os.path.join(MNT_PATH, 'screening', 'screening_results_long.json')
MAX_MODEL_LEN = 16384
MAX_NEW_TOKENS = 1024
CHUNK_SIZE_CHARS = 4000
OVERLAP_SENTENCES = 3

# ==========================================
# システムプロンプト: 外部知識の注入
# 「医療系全般」の知識として、ヘルスリテラシーの判断基準(かちもない)と
# フェイク定義を明文化して与える。
# ==========================================
with open('/home/wake/projects/proposal/local_llm/screening/prompt/system_prompt_v6.txt', encoding='utf-8') as f:
    SYSTEM_PROMPT_TEXT = f.read()

# ==========================================
# テキスト処理クラス
# ==========================================
class TextProcessor:
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """
        日本語テキストを文単位に分割する。
        「。」「！」「？」や改行を区切りとするが、括弧内は無視する等の処理を含む。
        """
        text = str(text).strip()
        # 肯定後読み(?<=...)で区切り文字を含めて分割
        # 否定先読み(?![...])で閉じ括弧の直前にある句点は区切りとみなさない（会話文対策）
        pattern = r'(?<=[\。\！\？\n])(?![」』\)])'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    @staticmethod
    def create_chunks(sentences: List[str], chunk_size: int, overlap: int) -> List:
        """
        スライディングウィンドウ方式でチャンクを作成する。
        """
        chunks = []
        current_chunk = []
        current_length = 0

        # イテレータ管理用
        idx = 0
        while idx < len(sentences):
            sentence = sentences[idx]
            current_chunk.append(sentence)
            current_length += len(sentence)

            # チャンクサイズ超過 または 最後の文
            if current_length >= chunk_size or idx == len(sentences) - 1:
                chunks.append({
                    "text": "".join(current_chunk),
                    "start_idx": max(0, idx - len(current_chunk) + 1), # 概算の開始位置
                })

                # オーバーラップ処理: 次のチャンクのためにインデックスを戻す
                # 現在位置からoverlap分だけ戻るが、進捗ゼロにならないように制御
                step_back = min(len(current_chunk) - 1, overlap)
                idx -= step_back

                # バッファリセット
                current_chunk = []
                current_length = 0

            idx += 1

        return chunks

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        検索用にテキストを正規化する（空白、改行、句読点を除去）
        """
        # 全角・半角スペース、改行、タブを除去
        text = re.sub(r'\s+', '', text)
        # 句読点、ダブルクォート(")、シングルクォート(')、バックスラッシュ(\)を除去
        text = re.sub(r'[、。！？\.\,\!\"\'\\]', '', text)
        return text

    @staticmethod
    def find_original_context_safe(all_sentences: List[str], quote: str, window: int = 2) -> Dict:
        """
        正規化マッチングを用いて原文から文脈を特定する
        """
        full_text = "".join(all_sentences)

        # 1. 完全一致 or 簡易検索
        if quote in full_text:
             pass

        norm_quote = TextProcessor.normalize_text(quote)
        if not norm_quote:
            return {"found": False, "context": "", "error": "quote_empty"}

        norm_sentences = [TextProcessor.normalize_text(s) for s in all_sentences]

        best_score = 0.0
        best_idx = -1

        # 全文を結合した上で正規化インデックスを作るマップを作成するのはコストが高いので
        # 各文ごとの類似度チェック（正規化済み）を行う

        # --- アプローチ1: 単文ごとのマッチング ---
        for i, norm_sent in enumerate(norm_sentences):
            if not norm_sent:
                continue

            # 完全包含チェック
            if norm_quote in norm_sent:
                best_score = 1.0
                best_idx = i
                break

            # Fuzzy Matching
            matcher = difflib.SequenceMatcher(None, norm_quote, norm_sent)
            score = matcher.ratio()
            if score > best_score:
                best_score = score
                best_idx = i

        # --- アプローチ2: 文またぎ（2文連結）への対応 ---
        if best_score < 0.95:
            for i in range(len(norm_sentences) - 1):
                merged_sent = norm_sentences[i] + norm_sentences[i+1]
                if not merged_sent:
                    continue

                if norm_quote in merged_sent:
                    best_score = 1.0
                    best_idx = i
                    break

                matcher = difflib.SequenceMatcher(None, norm_quote, merged_sent)
                score = matcher.ratio()

                if score > best_score:
                    best_score = score
                    best_idx = i

        # --- 結果判定 ---
        THRESHOLD = 0.65
        if best_idx != -1 and best_score >= THRESHOLD:
            # 2文連結も考慮して少し広めに取得 (+2)
            # quoteが1文の場合：前2文+quote+後ろ3文=6文
            # quoteが2文の場合（best_idxとbest_idx+1がquote）：前2文+quote2文+後ろ2文=6文
            start = max(0, best_idx - window)
            end = min(len(all_sentences), best_idx + window + 2)

            return {
                "found": True,
                "context": "".join(all_sentences[start:end]),
                "score": best_score,
                "error": None
            }
        else:
            return {
                "found": False,
                "context": "",
                "score": best_score,
                "error": "low_confidence_match"
            }

# ==========================================
# メイン・スクリーニングクラス
# ==========================================
class MedicalFakeScreener:
    def __init__(self, model_path: str):
        print(f"Loading Model: {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        ).eval()

    def generate_response(self, text_chunk: str) -> str:
        """
        Transformersを用いた推論実行
        """
        formatted_system_prompt = (
            f"{SYSTEM_PROMPT_TEXT}\n\n"
            "=========================================\n"
            "【分析対象テキスト】\n"
            "<TRANSCRIPT>\n"
            f"{text_chunk}\n"
            "</TRANSCRIPT>\n"
            "========================================="
        )

        messages = [
            {"role": "system", "content": formatted_system_prompt},
            {"role": "user", "content": "上記の <TRANSCRIPT> を分析し、抽出基準に従って結果を出力してください。"}
        ]

        # チャットテンプレートの適用
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            add_special_tokens=False
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,     # 再現性重視のためGreedy
                repetition_penalty=1.1
            )

        # 入力プロンプト部分を除去してデコード
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        return generated_text

    def parse_tagged_output(self, text: str) -> Dict:
        """
        タグ形式の出力をパースして辞書リストに変換する
        """
        candidates = []

        if "該当なし" in text or "見つかりません" in text:
            return {"candidates": [], "raw_output": text, "status": "success_no_candidates"}

        # <CANDIDATE>タグの中身をすべて抽出（改行コード含む）
        # re.DOTALL で . が改行にもマッチするようにする
        entry_pattern = r'<CANDIDATE>(.*?)</CANDIDATE>'
        entries = re.findall(entry_pattern, text, re.DOTALL)

        if not entries:
            # "該当なし" などの場合は空リストを返す
            return {"candidates": [], "raw_output": text, "status": "success_no_candidates"}

        for entry in entries:
            try:
                # 各フィールドの抽出
                quote_match = re.search(r'<QUOTE>(.*?)</QUOTE>', entry, re.DOTALL)
                type_match = re.search(r'<TYPE>(.*?)</TYPE>', entry, re.DOTALL)
                reason_match = re.search(r'<REASON>(.*?)</REASON>', entry, re.DOTALL)

                quote = quote_match.group(1).strip() if quote_match else ""

                ignore_keywords = ["チャンネル登録", "高評価", "こんにちは", "さようなら", "ゆっくりしていってね"]
                if any(kw in quote for kw in ignore_keywords):
                    continue

                type_val = type_match.group(1).strip() if type_match else "0"
                reason = reason_match.group(1).strip() if reason_match else ""

                # typeの数値変換（数字以外が入っていた場合のガード）
                type_id = int(re.sub(r'\D', '', type_val)) if re.search(r'\d', type_val) else 0

                if quote:
                    candidates.append({
                        "quote": quote,
                        "type": type_id,
                        "reason": reason
                    })
            except Exception as e:
                print(f"    Parse warning for entry: {e}")
                continue

        return {"candidates": candidates, "raw_output": text, "status": "success"}

    def process_dataset(self, df: pd.DataFrame) -> List:
        results = []

        for _, row in tqdm(df.iterrows(), total=len(df)):
            transcript = row['transcript']
            # メタデータがあればここで取得（例：動画IDなど）
            video_id = row.get('video_id')
            label = row.get('label')

            print(f"Processing Video ID: {video_id}...")

            # 1. 前処理 & 分割
            sentences = TextProcessor.split_into_sentences(transcript)
            chunks = TextProcessor.create_chunks(sentences, CHUNK_SIZE_CHARS, OVERLAP_SENTENCES)

            video_candidates = []

            # 2. チャンクごとに推論 (逐次処理)
            for i, chunk in enumerate(chunks):
                print(f"  - Chunk {i+1}/{len(chunks)} analyzing...")

                generated_text = self.generate_response(chunk['text'])
                data = self.parse_tagged_output(generated_text)

                candidates = data.get("candidates", [])

                if not isinstance(candidates, list):
                    continue

                for item in candidates:
                    quote = item.get("quote", "")
                    if not quote or len(quote) < 5:
                        continue

                    norm_quote = TextProcessor.normalize_text(quote)
                    ctx_result = TextProcessor.find_original_context_safe(sentences, norm_quote)
                    # 検索失敗時のハンドリング
                    if not ctx_result["found"]:
                        print(f"    -> Quote Not Found. Marking for Step 3.")
                        video_candidates.append({
                            "type": item.get("type"),
                            "reason": item.get("reason"),
                            "quote": quote,
                            "context_step2": chunk['text'], # チャンク全文で代用
                            "source_chunk_idx": i,
                            "lookup_error": "not_found_in_source",
                            "requires_full_context_check": True # Step 2 失敗
                        })

                    else:
                        # 正常検出
                        video_candidates.append({
                            "type": item.get("type"),
                            "reason": item.get("reason"),
                            "quote": quote,
                            "context_step2": ctx_result["context"],
                            "source_chunk_idx": i,
                            "requires_full_context_check": False # Step 2 成功
                        })

            results.append({
                "video_id": video_id,
                "label": label,
                "flagged_count": len(video_candidates),
                "candidates": video_candidates,
            })

        return results

# ==========================================
# 実行例
# ==========================================
if __name__ == "__main__":
    df = pd.read_csv(INPUT_CSV_PATH, dtype=str, encoding="utf-8")

    df = df[df['is_long'] == "True"].copy()

    screener = MedicalFakeScreener(MODEL_PATH)
    results = screener.process_dataset(df)

    with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)