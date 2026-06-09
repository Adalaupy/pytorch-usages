from pathlib import Path

import numpy as np
import pandas as pd

from use_case.Recommendation_System.data.prepare_data import user_events_df


USER_KEY_COL = "user_key"
ANIME_KEY_COL = "anime_key"
RATING_COL = "rating"
DEFAULT_PARALLEL_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "parallel_model_outputs"


# ================================================================================================
# Build and save shared user-anime signal tables for fast online recommendation
# ================================================================================================
def prepare_user_anime_signal_parquets(
    output_dir: str | Path | None = None,
    love_rating_threshold: float = 8.0,
) -> dict[str, Path]:
    output_path = Path(output_dir) if output_dir else DEFAULT_PARALLEL_OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    df_events = user_events_df()

    # Unique user-anime watch history for exclusion during recommendation serving.
    df_watch_history = df_events[[USER_KEY_COL, ANIME_KEY_COL]].dropna().drop_duplicates().copy()

    # Global watch popularity/confidence stats per anime.
    real_mask = df_events[RATING_COL].notna() & df_events[RATING_COL].ne(-1)
    real_events = df_events[real_mask]

    df_watch_stats = df_events.groupby(ANIME_KEY_COL, as_index=False).agg(
        n_watch_events=(ANIME_KEY_COL, "size"),
        n_watch_users=(USER_KEY_COL, "nunique"),
    )
    df_rating_counts = real_events.groupby(ANIME_KEY_COL, as_index=False).agg(
        n_real_ratings=(RATING_COL, "size"),
    )
    df_watch_stats = df_watch_stats.merge(df_rating_counts, on=ANIME_KEY_COL, how="left")
    df_watch_stats["n_real_ratings"] = df_watch_stats["n_real_ratings"].fillna(0).astype(int)
    df_watch_stats["rating_rate"] = (
        df_watch_stats["n_real_ratings"] / df_watch_stats["n_watch_events"]
    )

    # User loved anime derived from real ratings.
    df_user_loved = real_events[real_events[RATING_COL] >= love_rating_threshold].copy()
    df_user_loved = df_user_loved[[USER_KEY_COL, ANIME_KEY_COL, RATING_COL]]
    df_user_loved = df_user_loved.groupby([USER_KEY_COL, ANIME_KEY_COL], as_index=False).agg(
        loved_rating=(RATING_COL, "max"),
    )

    watch_path = output_path / "user_watch_history.parquet"
    watch_stats_path = output_path / "anime_watch_stats.parquet"
    loved_path = output_path / "user_loved_anime.parquet"

    df_watch_history.to_parquet(watch_path, index=False)
    df_watch_stats.to_parquet(watch_stats_path, index=False)
    df_user_loved.to_parquet(loved_path, index=False)

    return {
        "user_watch_history": watch_path,
        "anime_watch_stats": watch_stats_path,
        "user_loved_anime": loved_path,
    }


# ================================================================================================
# Recommend anime for target user based on similar users + precomputed shared parquets
# ================================================================================================
def recommend_anime_from_similar_users(
    target_user_key: int,
    df_similar_users: pd.DataFrame,
    output_dir: str | Path | None = None,
    top_k: int = 20,
    min_support_users: int = 2,
) -> pd.DataFrame:
    output_path = Path(output_dir) if output_dir else DEFAULT_PARALLEL_OUTPUT_DIR

    watch_path = output_path / "user_watch_history.parquet"
    watch_stats_path = output_path / "anime_watch_stats.parquet"
    loved_path = output_path / "user_loved_anime.parquet"

    for req_path in [watch_path, watch_stats_path, loved_path]:
        if not req_path.exists():
            raise FileNotFoundError(
                f"Required parquet not found at {req_path}. Run prepare_user_anime_signal_parquets first."
            )

    df_watch_history = pd.read_parquet(watch_path)
    df_watch_stats = pd.read_parquet(watch_stats_path)
    df_user_loved = pd.read_parquet(loved_path)

    similar_user_keys = set(df_similar_users[USER_KEY_COL].dropna().astype(int).tolist())
    if not similar_user_keys:
        return pd.DataFrame(
            columns=[ANIME_KEY_COL, "score", "support_user_count", "mean_loved_rating"]
        )

    watched_by_target = set(
        df_watch_history.loc[
            df_watch_history[USER_KEY_COL] == int(target_user_key),
            ANIME_KEY_COL,
        ].dropna().astype(int).tolist()
    )

    df_candidates = df_user_loved[df_user_loved[USER_KEY_COL].isin(similar_user_keys)].copy()
    df_candidates = df_candidates[~df_candidates[ANIME_KEY_COL].isin(watched_by_target)]
    if df_candidates.empty:
        return pd.DataFrame(
            columns=[ANIME_KEY_COL, "score", "support_user_count", "mean_loved_rating"]
        )

    df_agg = df_candidates.groupby(ANIME_KEY_COL, as_index=False).agg(
        support_user_count=(USER_KEY_COL, "nunique"),
        mean_loved_rating=("loved_rating", "mean"),
    )
    df_agg = df_agg[df_agg["support_user_count"] >= min_support_users].copy()
    if df_agg.empty:
        return pd.DataFrame(
            columns=[ANIME_KEY_COL, "score", "support_user_count", "mean_loved_rating"]
        )

    df_agg = df_agg.merge(
        df_watch_stats[[ANIME_KEY_COL, "n_watch_users"]],
        on=ANIME_KEY_COL,
        how="left",
    )
    df_agg["n_watch_users"] = df_agg["n_watch_users"].fillna(0)

    support_norm = df_agg["support_user_count"] / max(1, df_agg["support_user_count"].max())
    rating_norm = np.clip(df_agg["mean_loved_rating"] / 10.0, 0.0, 1.0)
    watch_conf = np.log1p(df_agg["n_watch_users"])
    watch_conf = watch_conf / max(1e-9, watch_conf.max())

    df_agg["score"] = 0.65 * support_norm + 0.25 * rating_norm + 0.10 * watch_conf
    out_cols = [ANIME_KEY_COL, "score", "support_user_count", "mean_loved_rating"]
    return df_agg[out_cols].sort_values("score", ascending=False).head(top_k).reset_index(drop=True)


# ================================================================================================
# Rank user's favorite anime from own history only (simple early-stage baseline)
# ================================================================================================
def rank_user_favorite_anime_simple(
    target_user_key: int,
    top_k: int = 10,
) -> pd.DataFrame:
    df_events = user_events_df()
    target_user_key = int(target_user_key)

    df_user = df_events[df_events[USER_KEY_COL] == target_user_key].copy()
    if df_user.empty:
        return pd.DataFrame(
            columns=[
                ANIME_KEY_COL,
                "score",
                "user_avg_rating",
                "user_watch_count",
                "same_anime_watch_users",
            ]
        )

    df_pop = df_events.groupby(ANIME_KEY_COL, as_index=False).agg(
        same_anime_watch_users=(USER_KEY_COL, "nunique"),
    )

    real_mask = df_user[RATING_COL].notna() & df_user[RATING_COL].ne(-1)
    df_user_real = df_user[real_mask]

    df_count = df_user.groupby(ANIME_KEY_COL, as_index=False).agg(
        user_watch_count=(ANIME_KEY_COL, "size"),
    )
    df_rating = df_user_real.groupby(ANIME_KEY_COL, as_index=False).agg(
        user_avg_rating=(RATING_COL, "mean"),
    )

    df_out = df_count.merge(df_rating, on=ANIME_KEY_COL, how="left")
    df_out = df_out.merge(df_pop, on=ANIME_KEY_COL, how="left")
    df_out["user_avg_rating"] = df_out["user_avg_rating"].fillna(0.0)
    df_out["same_anime_watch_users"] = df_out["same_anime_watch_users"].fillna(0).astype(int)

    rating_norm = np.clip(df_out["user_avg_rating"] / 10.0, 0.0, 1.0)

    watch_max = max(1, int(df_out["user_watch_count"].max()))
    watch_norm = df_out["user_watch_count"] / watch_max

    pop_log = np.log1p(df_out["same_anime_watch_users"])
    pop_norm = pop_log / max(1e-9, float(pop_log.max()))

    df_out["score"] = 0.60 * rating_norm + 0.30 * watch_norm + 0.10 * pop_norm

    out_cols = [
        ANIME_KEY_COL,
        "score",
        "user_avg_rating",
        "user_watch_count",
        "same_anime_watch_users",
    ]
    return (
        df_out[out_cols]
        .sort_values(
            by=["score", "user_avg_rating", "user_watch_count", ANIME_KEY_COL],
            ascending=[False, False, False, True],
        )
        .head(top_k)
        .reset_index(drop=True)
    )
