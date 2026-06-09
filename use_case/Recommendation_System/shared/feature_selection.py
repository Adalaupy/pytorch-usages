import numpy as np
import pandas as pd


# ================================================================================================
# Select numeric columns with correlation-based redundancy pruning
# ================================================================================================
def remove_correlated_features(
    df: pd.DataFrame,
    exclude_cols: list[str] | None = None,
    corr_threshold: float = 0.7,
) -> list[str]:
    exclude_set = set(exclude_cols or [])

    numeric_cols = [
        col
        for col in df.columns
        if col not in exclude_set and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not numeric_cols:
        print("[feature_selection] total_numeric_cols=0 removed_cols=0 remaining_cols=0")
        return []

    # Drop constant columns first (variance exactly zero).
    variances = df[numeric_cols].fillna(0.0).var(ddof=0)
    candidate_cols = [col for col in numeric_cols if variances[col] > 0]
    if not candidate_cols:
        print(
            f"[feature_selection] total_numeric_cols={len(numeric_cols)} "
            f"removed_cols={len(numeric_cols)} remaining_cols=0"
        )
        return []

    corr_abs = df[candidate_cols].corr().abs()
    upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))

    missing_rate = df[candidate_cols].isna().mean()
    variances = df[candidate_cols].fillna(0.0).var(ddof=0)

    drop_cols: set[str] = set()
    for col_i in candidate_cols:
        for col_j in candidate_cols:
            if col_i >= col_j:
                continue
            corr_val = upper.at[col_i, col_j]
            if pd.isna(corr_val) or corr_val <= corr_threshold:
                continue

            # Deterministic keep/drop rule:
            # 1) keep lower missing-rate column
            # 2) if tie, keep higher variance column
            # 3) if tie, keep lexical smaller column name
            miss_i = float(missing_rate[col_i])
            miss_j = float(missing_rate[col_j])
            var_i = float(variances[col_i])
            var_j = float(variances[col_j])

            if miss_i < miss_j:
                drop_cols.add(col_j)
            elif miss_j < miss_i:
                drop_cols.add(col_i)
            elif var_i > var_j:
                drop_cols.add(col_j)
            elif var_j > var_i:
                drop_cols.add(col_i)
            else:
                drop_cols.add(max(col_i, col_j))

    selected_cols = [col for col in candidate_cols if col not in drop_cols]
    selected_cols.sort()

    total_cols = len(numeric_cols)
    remaining_cols = len(selected_cols)
    removed_cols = total_cols - remaining_cols
    print(
        f"[feature_selection] total_numeric_cols={total_cols} "
        f"removed_cols={removed_cols} remaining_cols={remaining_cols}"
    )

    return selected_cols
