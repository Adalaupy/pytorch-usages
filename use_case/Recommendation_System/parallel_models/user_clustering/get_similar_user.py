from pathlib import Path

import pandas as pd


USER_KEY_COL = "user_key"
DEFAULT_PARALLEL_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "parallel_model_outputs"


# ================================================================================================
# Load precomputed user cluster list from parquet
# ================================================================================================
def load_user_cluster_list(cluster_parquet_path: str | Path | None = None) -> pd.DataFrame:

    path = Path(cluster_parquet_path) if cluster_parquet_path else DEFAULT_PARALLEL_OUTPUT_DIR / "user_cluster_list.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Cluster parquet not found at {path}. Run offline user clustering first."
        )
    return pd.read_parquet(path)


# ================================================================================================
# Get users in the same cluster as target user
# ================================================================================================
def main_get_similar_user(
    target_user_key: int,
    top_k: int = 10,
    cluster_parquet_path: str | Path | None = None,
) -> pd.DataFrame:

    df_cluster = load_user_cluster_list(cluster_parquet_path)
    target_user_key = int(target_user_key)

    target_row = df_cluster[df_cluster[USER_KEY_COL] == target_user_key]
    if target_row.empty:
        return pd.DataFrame(columns=[USER_KEY_COL, "cluster_id", "cluster_size"])

    target_cluster_id = int(target_row.iloc[0]["cluster_id"])

    out = df_cluster[
        (df_cluster["cluster_id"] == target_cluster_id)
        & (df_cluster[USER_KEY_COL] != target_user_key)
    ].copy()
    if out.empty:
        return pd.DataFrame(columns=[USER_KEY_COL, "cluster_id", "cluster_size"])

    out = out.sort_values(USER_KEY_COL, ascending=True).head(top_k).reset_index(drop=True)
    return out[[USER_KEY_COL, "cluster_id", "cluster_size"]]

