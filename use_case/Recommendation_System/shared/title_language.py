import re

import numpy as np
import pandas as pd


def normalize_title_text(title: str) -> str:
    # Normalize raw title text into a lowercase alphanumeric form for robust matching.
    text = str(title).lower().strip()
    text = re.sub(r"\[[^\]]*\]|\([^\)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_sequel_index(normalized_title: str) -> int:
    # Infer a sequel index from common numeric, season, part, and Roman numeral title patterns.
    patterns = [
        r"\bseason\s*(\d+)\b",
        r"\bpart\s*(\d+)\b",
        r"\b(\d+)(st|nd|rd|th)\s+season\b",
        r"\bs(\d+)\b",
        r"\bii\b",
        r"\biii\b",
        r"\biv\b",
        r"\bv\b",
        r"\b(\d+)\b$",
    ]

    for p in patterns:
        m = re.search(p, normalized_title)
        if not m:
            continue

        token = m.group(1) if m.groups() else ""
        if token.isdigit():
            return int(token)

        roman_map = {"ii": 2, "iii": 3, "iv": 4, "v": 5}
        roman_token = m.group(0).strip()
        if roman_token in roman_map:
            return roman_map[roman_token]

    return 1


def derive_series_key(title: str) -> str:
    # Strip sequel markers and release-format words to keep a franchise-style base series key.
    text = normalize_title_text(title)
    # Remove common sequel markers to form franchise/series base key.
    text = re.sub(r"\bseason\s*\d+\b", " ", text)
    text = re.sub(r"\bpart\s*\d+\b", " ", text)
    text = re.sub(r"\b\d+(st|nd|rd|th)\s+season\b", " ", text)
    text = re.sub(r"\bs\d+\b", " ", text)
    text = re.sub(r"\b(movie|ova|ona|special)\b", " ", text)
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\b(ii|iii|iv|v)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _safe_token(s: str) -> str:
    # Convert arbitrary text into a safe lowercase token for deterministic feature names.
    token = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return token or "unknown"


def build_title_series_matrix(
    df: pd.DataFrame,
    title_col: str = "anime_title_raw",
    min_series_size: int = 2,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    # Build series one-hot features plus normalized sequel index and return metadata for debugging.
    if title_col not in df.columns:
        return np.zeros((len(df), 0), dtype=float), [], pd.DataFrame(index=df.index)

    titles = df[title_col].fillna("unknown").astype(str)
    normalized = titles.map(normalize_title_text)
    series_keys = titles.map(derive_series_key)
    sequel_idx = normalized.map(extract_sequel_index).astype(float)

    series_counts = series_keys.value_counts()
    keep_series = sorted(
        [s for s, cnt in series_counts.items() if cnt >= min_series_size and s != ""]
    )

    series_matrix = np.zeros((len(df), len(keep_series)), dtype=float)
    series_to_col = {k: i for i, k in enumerate(keep_series)}
    for row_idx, s in enumerate(series_keys.tolist()):
        col_idx = series_to_col.get(s)
        if col_idx is not None:
            series_matrix[row_idx, col_idx] = 1.0

    sequel_norm = sequel_idx.to_numpy(dtype=float)
    sequel_norm = sequel_norm / max(1.0, float(sequel_norm.max()))
    sequel_norm = sequel_norm.reshape(-1, 1)

    if series_matrix.shape[1] > 0:
        X = np.concatenate([series_matrix, sequel_norm], axis=1)
    else:
        X = sequel_norm

    feature_names = [f"series_key_{_safe_token(s)}" for s in keep_series] + ["series_sequel_index_norm"]

    meta = pd.DataFrame(
        {
            "series_key": series_keys,
            "normalized_title": normalized,
            "series_sequel_index": sequel_idx,
        }
    )
    return X, feature_names, meta
