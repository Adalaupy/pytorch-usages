from pathlib import Path

import numpy as np
import pandas as pd

from use_case.Recommendation_System.data.prepare_data import (
    anime_content_ohe_df,
    anime_relative_stats_df,
    
)
from use_case.Recommendation_System.shared.clustering import (
    build_cluster_list_df,
    dynamic_cluster_count,
    kmeans_numpy,
    load_selected_feature_names,
    resolve_numeric_feature_cols,
    save_parquet_artifacts,
    standardize_matrix,
)
from use_case.Recommendation_System.shared.title_language import (
    build_title_series_matrix,
)


ANIME_KEY_COL = "anime_key"
DEFAULT_PARALLEL_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "parallel_model_outputs"
DEFAULT_CORR_THRESHOLD = 0.7


# ================================================================================================
# Build standardized anime feature matrix from content and relative stats
# ================================================================================================
def build_anime_feature_matrix(
    selected_feature_cols: list[str] | None = None,
    corr_threshold: float = DEFAULT_CORR_THRESHOLD,
    use_title_series_features: bool = True,
    title_series_weight: float = 0.35,
    min_series_size: int = 2,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    # Merge anime content and relative stats, resolve usable numeric features, and prepare the base matrix.
    df_anime_content = anime_content_ohe_df()
    df_anime_relative = anime_relative_stats_df()

    df_anime = df_anime_content.merge(df_anime_relative, on=ANIME_KEY_COL, how="inner")
    feature_cols = resolve_numeric_feature_cols(
        df=df_anime,
        key_col=ANIME_KEY_COL,
        selected_feature_cols=selected_feature_cols,
        corr_threshold=corr_threshold,
        empty_error_message="No numeric feature columns found for anime clustering.",
    )

    anime_keys = df_anime[ANIME_KEY_COL].astype(int).to_numpy()
    X_base = df_anime[feature_cols].fillna(0.0).to_numpy(dtype=float)
    final_feature_cols = list(feature_cols)

    # Optionally append weighted title-series features so sequels and franchise continuity influence clustering.
    if use_title_series_features:
        title_matrix, title_feature_names, _ = build_title_series_matrix(
            df=df_anime,
            title_col="anime_title_raw",
            min_series_size=min_series_size,
        )
        if title_matrix.shape[1] > 0:
            X = np.concatenate([X_base, title_series_weight * title_matrix], axis=1)  # Join base numeric features and title-series features column-wise.
            final_feature_cols.extend(title_feature_names)
        else:
            X = X_base
    else:
        X = X_base

    # Standardize feature scales before KMeans to prevent large-magnitude columns from dominating.
    Xs = standardize_matrix(X)

    return anime_keys, Xs, final_feature_cols, df_anime[[ANIME_KEY_COL]].copy()


# ================================================================================================
# Build whole anime-catalog clustering list
# ================================================================================================
def build_anime_cluster(
    n_clusters: int | None = None,
    max_iter: int = 100,
    random_state: int = 42,
    corr_threshold: float = DEFAULT_CORR_THRESHOLD,
    use_title_series_features: bool = True,
    title_series_weight: float = 0.35,
    min_series_size: int = 2,
    feature_bundle: tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    # Reuse a provided feature bundle when available, otherwise build one from current settings.
    if feature_bundle is None:
        anime_keys, X, _, _ = build_anime_feature_matrix(
            corr_threshold=corr_threshold,
            use_title_series_features=use_title_series_features,
            title_series_weight=title_series_weight,
            min_series_size=min_series_size,
        )
    else:
        anime_keys, X, _, _ = feature_bundle

    anime_keys = anime_keys.astype(int)
    cluster_count = n_clusters if n_clusters is not None else dynamic_cluster_count(len(anime_keys), min_clusters=2)

    # Run shared NumPy KMeans and package keys with cluster labels for downstream retrieval.
    labels = kmeans_numpy(
        X=X,
        n_clusters=cluster_count,
        max_iter=max_iter,
        random_state=random_state,
    )

    out = build_cluster_list_df(
        item_keys=anime_keys,
        labels=labels,
        key_col=ANIME_KEY_COL,
    )
    return out, anime_keys, X


# ================================================================================================
# Run full offline anime-clustering pipeline and save parquet outputs
# ================================================================================================
def main_run_anime_clustering_to_parquet(
    output_dir: str | Path | None = None,
    n_clusters: int | None = None,
    max_iter: int = 100,
    random_state: int = 42,
    corr_threshold: float = DEFAULT_CORR_THRESHOLD,
    use_title_series_features: bool = True,
    title_series_weight: float = 0.35,
    min_series_size: int = 2,
) -> dict[str, Path]:
    # Execute offline feature build and clustering, then persist grouping artifacts.
    output_path = Path(output_dir) if output_dir else DEFAULT_PARALLEL_OUTPUT_DIR
    output_path.mkdir(parents=True, exist_ok=True)

    selected_features_path = output_path / "anime_cluster_selected_features.parquet"

    selected_feature_cols = load_selected_feature_names(selected_features_path)

    feature_bundle = build_anime_feature_matrix(
        selected_feature_cols=selected_feature_cols,
        corr_threshold=corr_threshold,
        use_title_series_features=use_title_series_features,
        title_series_weight=title_series_weight,
        min_series_size=min_series_size,
    )
    _, _, final_feature_cols, _ = feature_bundle

    df_cluster_list, _, _ = build_anime_cluster(
        n_clusters=n_clusters,
        max_iter=max_iter,
        random_state=random_state,
        corr_threshold=corr_threshold,
        use_title_series_features=use_title_series_features,
        title_series_weight=title_series_weight,
        min_series_size=min_series_size,
        feature_bundle=feature_bundle,
    )

    return save_parquet_artifacts(
        output_dir=output_path,
        artifacts={
            "anime_cluster_list": df_cluster_list,
            "anime_cluster_selected_features": pd.DataFrame({"feature_name": final_feature_cols}),
        },
    )


if __name__ == "__main__":
    saved = main_run_anime_clustering_to_parquet()
    print(saved)
