from pathlib import Path

import pandas as pd

from use_case.Recommendation_System.data.prepare_data import (
    anime_relative_stats_df,
    user_events_df,
)
from use_case.Recommendation_System.shared.anime_interest import (
    rank_user_favorite_anime_simple,
)


USER_KEY_COL = "user_key"
ANIME_KEY_COL = "anime_key"
DEFAULT_PARALLEL_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "parallel_model_outputs"


def load_anime_cluster_list(cluster_parquet_path: str | Path | None = None) -> pd.DataFrame:
    # Resolve the anime-cluster artifact path using an optional override and a stable default location.
    path = (
        Path(cluster_parquet_path)
        if cluster_parquet_path
        else DEFAULT_PARALLEL_OUTPUT_DIR / "anime_cluster_list.parquet"
    )
    # Raise early when anime clusters are missing because this online step depends on grouped anime.
    if not path.exists():
        raise FileNotFoundError(
            f"Anime cluster parquet not found at {path}. Run offline anime clustering first."
        )
    return pd.read_parquet(path)


def _user_favorite_anime(
    target_user_key: int,
    top_n_favorite_anime: int = 5,
) -> list[int]:
    # Reuse shared favorite-anime ranking so favorite selection logic stays consistent across pipelines.
    df_favorite = rank_user_favorite_anime_simple(
        target_user_key=target_user_key,
        top_k=top_n_favorite_anime,
    )
    if df_favorite.empty:
        return []
    return df_favorite[ANIME_KEY_COL].dropna().astype(int).tolist()


# ================================================================================================
# Simple online suggestion: user favorite anime -> same-cluster anime suggestions
# ================================================================================================
def main_get_anime_cluster_suggestions(
    target_user_key: int,
    top_k: int = 20,
    top_n_favorite_anime: int = 5,
    cluster_parquet_path: str | Path | None = None,
) -> pd.DataFrame:
    # Load anime groups, user history, and anime quality stats required for simple same-group suggestion.
    df_cluster = load_anime_cluster_list(cluster_parquet_path)
    df_events = user_events_df()
    df_anime_rel = anime_relative_stats_df()

    # Step 1 gets the user's favorite anime list as anchors for group-based suggestion.
    favorite_anime = _user_favorite_anime(
        target_user_key=target_user_key,
        top_n_favorite_anime=top_n_favorite_anime,
    )
    if not favorite_anime:
        return pd.DataFrame(
            columns=[
                ANIME_KEY_COL,
                "model2_score",
                "favorite_overlap_count",
                "quality_prior",
            ]
        )

    # Step 2 removes already watched anime so suggestions only include unseen anime.
    watched = set(
        df_events.loc[df_events[USER_KEY_COL] == int(target_user_key), ANIME_KEY_COL]
        .dropna()
        .astype(int)
        .tolist()
    )

    # Step 3 maps user's favorite anime to their clusters so we can find other anime in those clusters.
    df_favorite_cluster = df_cluster[df_cluster[ANIME_KEY_COL].isin(favorite_anime)][
        [ANIME_KEY_COL, "cluster_id"]
    ].rename(columns={ANIME_KEY_COL: "favorite_anime_key"})
    if df_favorite_cluster.empty:
        return pd.DataFrame(
            columns=[
                ANIME_KEY_COL,
                "model2_score",
                "favorite_overlap_count",
                "quality_prior",
            ]
        )

    # Step 4 finds same-cluster anime and counts how many favorite-anime links each anime receives.
    df_suggest = df_cluster[[ANIME_KEY_COL, "cluster_id"]].merge(
        df_favorite_cluster,
        on="cluster_id",
        how="inner",
    )
    df_suggest = df_suggest[df_suggest[ANIME_KEY_COL] != df_suggest["favorite_anime_key"]]

    df_agg = df_suggest.groupby(ANIME_KEY_COL, as_index=False).agg(
        favorite_overlap_count=("favorite_anime_key", "size"),
    )
    df_agg = df_agg[~df_agg[ANIME_KEY_COL].isin(watched)].copy()
    if df_agg.empty:
        return pd.DataFrame(
            columns=[
                ANIME_KEY_COL,
                "model2_score",
                "favorite_overlap_count",
                "quality_prior",
            ]
        )

    # Step 5 merges a quality prior that blends normalized z-score and confidence weight.
    df_quality = df_anime_rel[[ANIME_KEY_COL, "zscore_vs_global", "confidence_weight"]].copy()
    z = df_quality["zscore_vs_global"].fillna(0.0)
    z_norm = (z - z.min()) / max(1e-9, (z.max() - z.min()))
    df_quality["quality_prior"] = 0.5 * z_norm + 0.5 * df_quality["confidence_weight"].fillna(0.0)
    df_agg = df_agg.merge(df_quality[[ANIME_KEY_COL, "quality_prior"]], on=ANIME_KEY_COL, how="left")

    df_agg["quality_prior"] = df_agg["quality_prior"].fillna(0.0)

    overlap = df_agg["favorite_overlap_count"].fillna(0.0).astype(float)
    overlap_norm = overlap / max(1.0, float(overlap.max()))

    # Compute a simple score from same-cluster overlap and quality prior.
    df_agg["model2_score"] = 0.80 * overlap_norm + 0.20 * df_agg["quality_prior"]

    out_cols = [
        ANIME_KEY_COL,
        "model2_score",
        "favorite_overlap_count",
        "quality_prior",
    ]
    # Return deterministic top-k output sorted by score and anime key.
    return (
        df_agg[out_cols]
        .sort_values(["model2_score", "favorite_overlap_count", ANIME_KEY_COL], ascending=[False, False, True])
        .head(top_k)
        .reset_index(drop=True)
    )
