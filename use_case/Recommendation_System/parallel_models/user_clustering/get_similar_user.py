from pathlib import Path

import pandas as pd

from use_case.Recommendation_System.shared.anime_interest import (
    prepare_user_anime_signal_parquets,
    recommend_anime_from_similar_users,
)


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


# ================================================================================================
# Recommend anime for target user using similar users and shared precomputed parquets
# ================================================================================================
def main_recommend_anime_from_similar_user(
    target_user_key: int,
    top_k_similar_users: int = 30,
    top_k_anime: int = 20,
    min_support_users: int = 2,
    cluster_parquet_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    df_similar = main_get_similar_user(
        target_user_key=target_user_key,
        top_k=top_k_similar_users,
        cluster_parquet_path=cluster_parquet_path,
    )
    return recommend_anime_from_similar_users(
        target_user_key=target_user_key,
        df_similar_users=df_similar,
        output_dir=output_dir,
        top_k=top_k_anime,
        min_support_users=min_support_users,
    )


# ================================================================================================
# Offline utility to prepare shared recommendation signal parquets for fast serving
# ================================================================================================
def main_prepare_recommendation_signal_parquets(
    output_dir: str | Path | None = None,
    love_rating_threshold: float = 8.0,
) -> dict[str, Path]:
    return prepare_user_anime_signal_parquets(
        output_dir=output_dir,
        love_rating_threshold=love_rating_threshold,
    )

