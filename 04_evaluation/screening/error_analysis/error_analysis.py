import pandas as pd
import os
import json

MNT_PATH = os.getenv('MNT_PATH') or ''
DETAILS_PATH = os.path.join(MNT_PATH, "screening", "final_classification_details.csv")
STEP1_PATH = os.path.join(MNT_PATH, "screening", "screening_results.json")
OUTPUT_ERROR_CSV = "/home/wake/projects/proposal/local_llm/screening/error_analysis/final_error_analysis.csv"

def export_errors_for_analysis():
    if not os.path.exists(DETAILS_PATH):
        print("Details file not found.")
        return

    df = pd.read_csv(DETAILS_PATH)
    
    # Step 1の抽出テキストも紐付ける
    with open(STEP1_PATH, 'r', encoding='utf-8') as f:
        step1_data = {str(item['video_id']).strip(): item for item in json.load(f)}

    # エラーデータの抽出（False Negative と False Positive）
    errors = df[df['ground_truth'] != df['prediction']].copy()
    
    print(f"Total Errors: {len(errors)}")
    print(f" - False Negatives (Missed Fake): {len(errors[errors['ground_truth']=='fake'])}")
    print(f" - False Positives (False Alarm): {len(errors[errors['ground_truth']=='real'])}")

    # 分析用データの作成
    analysis_data = []
    for _, row in errors.iterrows():
        vid = str(row['video_id']).strip()
        item = step1_data.get(vid, {})
        candidates = item.get('candidates', [])
        
        # 代表的な抽出テキスト（最初の1つ）を取得
        extracted_text = ""
        if candidates:
            extracted_text = candidates[0].get('quote', '')

        analysis_data.append({
            "video_id": vid,
            "ground_truth": row['ground_truth'],
            "prediction": row['prediction'],
            "logic_type": row['logic_type'],
            "step1_flagged_count": row['step1_flagged_count'],
            "extracted_snippet": extracted_text[:200]  # 長すぎるのでカット
        })

    # 保存
    out_df = pd.DataFrame(analysis_data)
    out_df.to_csv(OUTPUT_ERROR_CSV, index=False, encoding='utf-8')
    print(f"Error analysis file saved to: {OUTPUT_ERROR_CSV}")

if __name__ == "__main__":
    export_errors_for_analysis()