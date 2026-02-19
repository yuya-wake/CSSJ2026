"""
Utility functions for Japanese LIWC-style lexical analysis (J-LIWC2015).

This module provides:
  - Tokenization via MeCab (wakati-gaki / whitespace-separated tokens)
  - Parsing of the J-LIWC2015 dictionary file (LIWC-like format with '%' section delimiters)
  - Computing per-category LIWC scores as percentages of tokens

Design notes
------------
- Environment-dependent settings (e.g., MeCab dictionary paths) must be provided by the caller.
  This keeps imports side-effect-free and avoids crashes when paths differ across machines.
- The J-LIWC2015 dictionary's lexical patterns may contain spaces (e.g., Japanese emoticons),
  so we must NOT split lexical lines by whitespace naively. We instead parse the trailing
  numeric category ID list from the end of each line.
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr


def wakati(text: str, mecab_cmd: List[str]) -> List[str]:
    """
    Tokenize Japanese text into whitespace-separated tokens using MeCab.

    Parameters
    ----------
    text : str
        Input text. If None/empty, returns [].
    mecab_cmd : list[str]
        Command arguments to invoke MeCab, e.g.:
          ["mecab", "-Owakati", "-d", IPADIC_DIR, "-u", USER_DIC]

    Returns
    -------
    list[str]
        List of tokens (strings) produced by MeCab.

    Raises
    ------
    RuntimeError
        If MeCab returns a non-zero exit code.
    """
    if text is None:
        return []

    text = str(text).strip()
    if not text:
        return []

    # Run MeCab as a subprocess. We pass input via stdin and capture stdout/stderr.
    p = subprocess.run(
        mecab_cmd,
        input=text + "\n",
        text=True,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if p.returncode != 0:
        # stderr typically includes diagnostic messages from MeCab.
        raise RuntimeError(f"MeCab failed:\n{p.stderr}")

    # "-Owakati" outputs tokens separated by spaces.
    return p.stdout.strip().split()


def load_liwc_dic(dic_path: str) -> Tuple[Dict[int, str], List[Tuple[str, Set[int]]]]:
    """
    Parse a J-LIWC2015 dictionary file.

    Expected file structure (LIWC-like):
      %                   <-- section delimiter
      <cat_id> <cat_name> <-- category definitions (one per line)
      ...
      %                   <-- section delimiter
      <pattern> <cat_id> <cat_id> ...  <-- lexical rules (one per line)

    Important:
      - Lexical <pattern> can contain spaces (e.g., emoticons). Therefore, we cannot
        simply split the entire line by whitespace. Instead, we parse the trailing
        numeric ID list at the end of the line and treat the left side as the pattern.

    Parameters
    ----------
    dic_path : str
        Path to "Japanese_Dictionary.dic" (J-LIWC2015).

    Returns
    -------
    cat_id2name : dict[int, str]
        Maps category ID -> category name (e.g., 54 -> "tentat").
    lexicon : list[tuple[str, set[int]]]
        List of (pattern, category_id_set). Patterns may end with '*' meaning prefix match.

    Raises
    ------
    ValueError
        If the dictionary does not contain at least two '%' delimiters, or parsing fails.
    """
    lines = Path(dic_path).read_text(encoding="utf-8", errors="ignore").splitlines()

    cat_id2name: Dict[int, str] = {}
    lexicon: List[Tuple[str, Set[int]]] = []

    # We switch parsing mode when encountering '%' lines.
    # First '%' starts category definitions; second '%' starts lexical rules.
    section = None  # None -> "cats" -> "words"
    pct = 0

    for raw in lines:
        s = raw.strip()
        if not s:
            continue

        if s == "%":
            pct += 1
            section = "cats" if pct == 1 else "words"
            continue

        if section == "cats":
            # Category definition line example:
            #   54      tentat
            parts = re.split(r"\s+", s, maxsplit=1)
            if len(parts) == 2 and re.fullmatch(r"\d+", parts[0]):
                cat_id2name[int(parts[0])] = parts[1].strip()
            # If a line doesn't match the expected format, we ignore it.
            continue

        if section == "words":
            # Lexical rule line example (pattern may contain spaces):
            #   （ ＾ω＾）*     30  31  120  122
            #
            # Strategy:
            #   - Capture the left "pattern" part lazily (.*?)
            #   - Capture the trailing numeric list:  \d+( \d+)*
            m = re.match(r"^(.*?)[\t ]+(\d+(?:[\t ]+\d+)*)\s*$", s)
            if not m:
                # Ignore lines that don't match lexical format (rare)
                continue

            pat = m.group(1).strip()
            # Extract all integers from the numeric tail.
            cids = set(map(int, re.findall(r"\d+", m.group(2))))
            if pat and cids:
                lexicon.append((pat, cids))
            continue

        # If section is still None, we ignore content before the first '%'.

    if pct < 2:
        raise ValueError("The dictionary must contain at least two '%' delimiters (cats/words).")
    if not cat_id2name or not lexicon:
        raise ValueError("Parsed dictionary is empty. Check file format/encoding.")

    return cat_id2name, lexicon


def match_token_to_pattern(token: str, pattern: str) -> bool:
    """
    Check whether a token matches a LIWC pattern.

    Rules:
      - If pattern ends with '*', perform prefix match.
      - Otherwise, require exact match.

    Parameters
    ----------
    token : str
        Token to test.
    pattern : str
        LIWC dictionary pattern (may end with '*').

    Returns
    -------
    bool
        True if token matches the pattern.
    """
    if pattern.endswith("*"):
        return token.startswith(pattern[:-1])
    return token == pattern


def liwc_scores(
    tokens: List[str],
    cat_id2name: Dict[int, str],
    lexicon: List[Tuple[str, Set[int]]],
) -> Dict[str, float]:
    """
    Compute LIWC category scores (% of tokens) for a token list.

    Notes
    -----
    - A single token may contribute to multiple categories (as in LIWC).
      Therefore, sums over categories may exceed 100%.
    - Scores are percentages: (hits_in_category / total_tokens) * 100.

    Parameters
    ----------
    tokens : list[str]
        Tokenized text.
    cat_id2name : dict[int, str]
        Category ID -> name mapping.
    lexicon : list[tuple[str, set[int]]]
        List of (pattern, category_id_set) rules.

    Returns
    -------
    dict[str, float]
        Category name -> percentage score.
    """
    total = len(tokens)

    # Initialize output with all categories (stable column set for DataFrame).
    out = {cat_id2name[cid]: 0 for cid in cat_id2name}
    if total == 0:
        return {k: 0.0 for k in out}

    # Split lexicon into exact-match rules and wildcard (prefix) rules.
    # This is a speed optimization: exact lookup is O(1) per token.
    exact: Dict[str, Set[int]] = {}
    wildcard: List[Tuple[str, Set[int]]] = []
    for pat, cids in lexicon:
        if pat.endswith("*"):
            wildcard.append((pat, cids))
        else:
            exact[pat] = exact.get(pat, set()) | cids

    # Count category hits in terms of token occurrences.
    hit = {cid: 0 for cid in cat_id2name}

    for tok in tokens:
        # Exact matches
        if tok in exact:
            for cid in exact[tok]:
                if cid in hit:
                    hit[cid] += 1

        # Wildcard matches (prefix). Assumed to be fewer than exact rules.
        for pat, cids in wildcard:
            if match_token_to_pattern(tok, pat):
                for cid in cids:
                    if cid in hit:
                        hit[cid] += 1

    # Convert counts to percentages.
    return {cat_id2name[cid]: 100.0 * hit[cid] / total for cid in hit}


def corr_table(
    out: pd.DataFrame,
    feats: list[str],
    method: Literal["spearman", "pearson"] = "spearman",
    x_col: str = "delta_len",
    y_prefix: str = "delta_",
    min_n: int = 3,
) -> pd.DataFrame:
    """
    Compute correlation between length change and feature change.

    This function measures how strongly feature changes (delta_<feature>) are associated with
    length changes (delta_len), using either Spearman or Pearson correlation.

    Parameters
    ----------
    out : pd.DataFrame
        DataFrame that contains:
          - x_col (default: "delta_len")
          - y_prefix + feature columns for each feature in `feats`
            (default prefix: "delta_")
    feats : list[str]
        Feature names without prefix, e.g., ["certain", "tentat"].
    method : {"spearman", "pearson"}, default "spearman"
        - "spearman": rank correlation (robust to monotonic non-linear relations)
        - "pearson" : linear correlation
    x_col : str, default "delta_len"
        Column name for x (predictor), usually the length difference.
    y_prefix : str, default "delta_"
        Prefix used for y columns (feature deltas).
    min_n : int, default 3
        Minimum number of valid (finite) pairs required to compute correlation.

    Returns
    -------
    pd.DataFrame
        Columns:
          - feature: feature name (without prefix)
          - corr: correlation coefficient
          - p: p-value
          - n: number of valid pairs used
          - median_delta: median of y among valid pairs
        Sorted by ascending p-value.

    Notes
    -----
    - NaN/inf are removed pairwise for each feature.
    - Features with fewer than `min_n` valid pairs are skipped.
    """
    if x_col not in out.columns:
        raise KeyError(f"corr_table: required column '{x_col}' not found in DataFrame.")

    rows: list[dict] = []
    x = out[x_col].to_numpy()

    for f in feats:
        y_col = f"{y_prefix}{f}"
        if y_col not in out.columns:
            # Skip silently to keep analysis robust across different dictionaries/features.
            continue

        y = out[y_col].to_numpy()
        ok = np.isfinite(x) & np.isfinite(y)

        if int(ok.sum()) < min_n:
            continue

        if method == "spearman":
            r, p = spearmanr(x[ok], y[ok])
        else:
            r, p = pearsonr(x[ok], y[ok])

        rows.append(
            {
                "feature": f,
                "corr": float(r),
                "p": float(p),
                "n": int(ok.sum()),
                "median_delta": float(np.median(y[ok])),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["feature", "corr", "p", "n", "median_delta"])

    return pd.DataFrame(rows).sort_values("p").reset_index(drop=True)


def residualize(y, x, min_n: int = 3):
    """
    Residualize y with respect to x using ordinary least squares (OLS).

    We fit the linear model:
        y = a + b*x + e
    and return residuals e (i.e., y with the linear effect of x removed).

    Parameters
    ----------
    y : array-like
        Response variable (e.g., delta_<feature>).
    x : array-like
        Predictor variable (e.g., delta_len).
    min_n : int, default 3
        Minimum number of valid pairs required to fit the model.

    Returns
    -------
    e : np.ndarray
        Residuals for the valid (finite) pairs only.
    beta : np.ndarray, shape (2,)
        Estimated coefficients [a, b] (intercept and slope).
    n_ok : int
        Number of valid pairs used.

    Notes
    -----
    - NaN/inf are removed pairwise.
    - Residuals correspond only to the filtered subset (ok == True).
    - If n_ok < min_n, a ValueError is raised to avoid unstable fits.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    ok = np.isfinite(x) & np.isfinite(y)
    x2 = x[ok]
    y2 = y[ok]

    n_ok = int(ok.sum())
    if n_ok < min_n:
        raise ValueError(f"residualize: not enough valid pairs (n_ok={n_ok} < {min_n}).")

    X = np.column_stack([np.ones_like(x2), x2])
    beta, *_ = np.linalg.lstsq(X, y2, rcond=None)
    y_hat = X @ beta
    e = y2 - y_hat

    return e, beta, n_ok


def scatter_with_fit(
    out: pd.DataFrame,
    feature: str,
    method_label: str,
    results_dir: str,
    x_col: str = "delta_len",
    y_prefix: str = "delta_",
    filename: str | None = None,
    min_n: int = 3,
) -> str | None:
    """
    Create a scatter plot of delta_<feature> vs delta_len with an OLS fitted line, and save it.

    This visualization is typically used to inspect whether feature changes are associated
    with length changes between baseline and generated text.

    Parameters
    ----------
    out : pd.DataFrame
        DataFrame containing x_col (default: "delta_len") and y_prefix+feature (default: "delta_<feature>").
    feature : str
        Feature name without prefix (e.g., "certain", "tentat").
    method_label : str
        Label for the compared method, used only for axis label text (e.g., "summary", "rag", "screening").
    results_dir : str
        Base directory to save plots. The function will create a subdirectory "{results_dir}/plt".
    x_col : str, default "delta_len"
        Column name for x.
    y_prefix : str, default "delta_"
        Prefix for y columns.
    filename : str | None, default None
        Output filename (without directory). If None, uses f"{feature}.png".
    min_n : int, default 3
        Minimum number of valid (finite) points required to generate a plot.

    Returns
    -------
    str | None
        Path to the saved image file, or None if the plot was skipped due to insufficient data.

    Notes
    -----
    - NaN/inf values are removed pairwise.
    - The fitted line is computed by ordinary least squares: y = a + b*x.
    """
    if x_col not in out.columns:
        raise KeyError(f"scatter_with_fit: required column '{x_col}' not found in DataFrame.")

    y_col = f"{y_prefix}{feature}"
    if y_col not in out.columns:
        # Feature column not available; skip silently for robustness.
        return None

    x = out[x_col].to_numpy()
    y = out[y_col].to_numpy()

    ok = np.isfinite(x) & np.isfinite(y)
    if int(ok.sum()) < min_n:
        return None

    x2, y2 = x[ok], y[ok]

    # Create plot
    plt.figure()
    plt.scatter(x2, y2)
    plt.xlabel(f"{x_col} ({method_label} - Baseline)")
    plt.ylabel(y_col)
    plt.title(f"{y_col} vs {x_col}")

    # OLS fit line: y = a + b*x
    X = np.column_stack([np.ones_like(x2), x2])
    beta, *_ = np.linalg.lstsq(X, y2, rcond=None)
    xs = np.linspace(x2.min(), x2.max(), 100)
    ys = beta[0] + beta[1] * xs
    plt.plot(xs, ys)

    # Save
    plt_dir = os.path.join(results_dir, "plt")
    os.makedirs(plt_dir, exist_ok=True)

    if filename is None:
        filename = f"{feature}.png"

    out_path = os.path.join(plt_dir, filename)
    plt.savefig(out_path)
    plt.close()

    return out_path