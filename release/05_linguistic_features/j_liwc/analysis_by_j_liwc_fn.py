import subprocess
import re
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, spearmanr, pearsonr
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

from utils import wakati, load_liwc_dic, liwc_scores, corr_table, residualize, scatter_with_fit

IPADIC_DIR = "/path/to/local/mecab/lib/mecab/dic/ipadic"
USER_DIC   = "/path/to/dic/j-liwc2015/user_dict.dic"
DIC_PATH = "/path/to/dic/j-liwc2015/Japanese_Dictionary.dic"
MECAB_CMD  = ["mecab", "-Owakati", "-d", IPADIC_DIR, "-u", USER_DIC]

MNT_PATH = os.getenv("MNT_PATH") or ""
summary_csv_path = os.path.join(MNT_PATH, "linguistic_features", "missed_by_summary_fn.csv")
rag_csv_path = os.path.join(MNT_PATH, "linguistic_features", "missed_by_rag_fn.csv")
screening_csv_path = os.path.join(MNT_PATH, "linguistic_features", "missed_by_screening_fn.csv")


csv_list = [summary_csv_path, rag_csv_path, screening_csv_path]
methods = ["summary", "rag", "screening"]


cat_id2name, lexicon = load_liwc_dic(DIC_PATH)

for method, csv_path in zip(methods, csv_list):

    df = pd.read_csv(csv_path, encoding="utf-8")

    base_rows, method_rows = [], []
    len_base_list, len_method_list = [], []

    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"Processing Method: {method}"):
        base_tokens = wakati(r["transcript"], MECAB_CMD)
        method_tokens  = wakati(r["text"], MECAB_CMD)

        base_rows.append(liwc_scores(base_tokens, cat_id2name, lexicon))
        method_rows.append(liwc_scores(method_tokens,  cat_id2name, lexicon))

        len_base_list.append(len(base_tokens))
        len_method_list.append(len(method_tokens))

    base_df = pd.DataFrame(base_rows).add_prefix("base_")
    method_df  = pd.DataFrame(method_rows).add_prefix(f"{method}_")
    out = pd.concat([df.reset_index(drop=True), base_df, method_df], axis=1)

    for cname in cat_id2name.values():
        out[f"delta_{cname}"] = out[f"{method}_{cname}"] - out[f"base_{cname}"]

    out["len_base"] = len_base_list
    out[f"len_{method}"] = len_method_list
    out["delta_len"] = out[f"len_{method}"] - out["len_base"]

    d_certain = out["delta_certain"].to_numpy()
    stat, p = wilcoxon(d_certain, alternative="less")
    print("delta_certain: p=", p, "median=", float(np.median(d_certain)))

    d_tentat = out["delta_tentat"].to_numpy()
    stat2, p2 = wilcoxon(d_tentat, alternative="greater")
    print("delta_tentat: p=", p2, "median=", float(np.median(d_tentat)))

    delta_cols = [c for c in out.columns if c.startswith("delta_") and c != "delta_len"]

    rows = []
    for col in delta_cols:
        x = out[col].to_numpy()
        x = x[np.isfinite(x)]
        # 全部同じ値（差分ゼロなど）の場合、wilcoxonがエラー/無意味になるのでp=1扱い
        if np.allclose(x, x[0]):
            stat, p = np.nan, 1.0
        else:
            try:
                # 探索目的なので two-sided が基本
                # zero_method="wilcox" はゼロ差分の扱いを明示（scipyの挙動差対策）
                stat, p = wilcoxon(x, alternative="two-sided", zero_method="wilcox")
            except ValueError:
                stat, p = np.nan, 1.0

        rows.append({
            "feature": col.replace("delta_", ""),
            "median_delta": float(np.median(x)) if len(x) else np.nan,
            "mean_delta": float(np.mean(x)) if len(x) else np.nan,
            "n": int(len(x)),
            "p_raw": float(p),
        })

    res = pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)

    # --- 多重補正（FDR: Benjamini-Hochberg）---
    reject, p_fdr, _, _ = multipletests(res["p_raw"].to_numpy(), alpha=0.05, method="fdr_bh")
    res["p_fdr_bh"] = p_fdr
    res["sig_fdr_0.05"] = reject

    sig = res[res["sig_fdr_0.05"]].copy()

    results_dir = f"results_{method}"
    os.makedirs(results_dir, exist_ok=True)
    res_no_len = res[res["feature"] != "len"].copy()
    sig_no_len = sig[sig["feature"] != "len"].copy()

    res_no_len.to_csv(os.path.join(results_dir, f"{method}_miss_wilcoxon_all_features_fdr_bh.csv"), index=False, encoding="utf-8")
    sig_no_len.to_csv(os.path.join(results_dir, f"{method}_miss_wilcoxon_significant_only_fdr_bh.csv"), index=False, encoding="utf-8")

    delta_cols = [c for c in out.columns if c.startswith("delta_")]
    keep_cols = ["video_id"] if "video_id" in out.columns else []
    keep_cols += delta_cols

    out[keep_cols].to_csv(os.path.join(results_dir, f"{method}_miss_delta_per_video.csv"), index=False, encoding="utf-8")

    sig_feats = sig_no_len["feature"].tolist()

    corr_s = corr_table(out, sig_feats, method="spearman")
    corr_p = corr_table(out, sig_feats, method="pearson")


    rows = []
    x = out["delta_len"].to_numpy()

    for f in sig_feats:
        y = out[f"delta_{f}"].to_numpy()
        e, beta, n_ok = residualize(y, x)

        # 残差の中央値が0からズレるか（two-sided）
        if np.allclose(e, e[0]):
            p = 1.0
        else:
            stat, p = wilcoxon(e, alternative="two-sided", zero_method="wilcox")

        rows.append({
            "feature": f,
            "n": n_ok,
            "beta_intercept": float(beta[0]),
            "beta_delta_len": float(beta[1]),
            "median_resid": float(np.median(e)),
            "p_resid_vs0": float(p),
        })

    resid_test = pd.DataFrame(rows).sort_values("p_resid_vs0").reset_index(drop=True)

    for f in sig_feats:
        scatter_with_fit(out, f, method, results_dir)