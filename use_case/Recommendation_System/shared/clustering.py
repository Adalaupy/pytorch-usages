import numpy as np
import pandas as pd
from pathlib import Path
from use_case.Recommendation_System.shared.feature_selection import remove_correlated_features


# ================================================================================================
# Run a lightweight k-means clustering with numpy
# ================================================================================================
def kmeans_numpy(
    X: np.ndarray,
    n_clusters: int,
    max_iter: int = 100,
    random_state: int = 42,
) -> np.ndarray:
    n_samples = X.shape[0]
    if n_samples == 0:
        return np.array([], dtype=int)

    n_clusters = max(1, min(n_clusters, n_samples))
    rng = np.random.default_rng(random_state)
    centroid_idx = rng.choice(n_samples, size=n_clusters, replace=False)
    centroids = X[centroid_idx].copy()

    labels = np.zeros(n_samples, dtype=int)
    for _ in range(max_iter):
        dist_sq = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(dist_sq, axis=1)

        if np.array_equal(labels, new_labels):
            break
        labels = new_labels

        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            if not np.any(mask):
                centroids[cluster_id] = X[rng.integers(0, n_samples)]
                continue
            centroids[cluster_id] = X[mask].mean(axis=0)

    return labels


# ================================================================================================
# Standardize numeric matrix columns with stable zero-variance handling
# ================================================================================================
def standardize_matrix(X: np.ndarray) -> np.ndarray:
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std


# ================================================================================================
# Dynamic cluster count equation: k = round(sqrt(N)) with minimum guard
# ================================================================================================
def dynamic_cluster_count(n_items: int, min_clusters: int = 2) -> int:
    return max(min_clusters, int(round(np.sqrt(max(1, n_items)))))


# ================================================================================================
# Read selected feature names parquet if present
# ================================================================================================
def load_selected_feature_names(path: str | Path, col_name: str = "feature_name") -> list[str] | None:
    path_obj = Path(path)
    if not path_obj.exists():
        return None

    df_selected = pd.read_parquet(path_obj)
    if col_name not in df_selected.columns:
        return None

    return df_selected[col_name].dropna().astype(str).tolist()


# ================================================================================================
# Save selected feature names parquet
# ================================================================================================
def save_selected_feature_names(path: str | Path, feature_names: list[str], col_name: str = "feature_name") -> None:
    pd.DataFrame({col_name: feature_names}).to_parquet(Path(path), index=False)


# ================================================================================================
# Save multiple dataframes as parquet artifacts in one output folder
# ================================================================================================
def save_parquet_artifacts(
    output_dir: str | Path,
    artifacts: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved: dict[str, Path] = {}
    for artifact_name, df in artifacts.items():
        path = output_path / f"{artifact_name}.parquet"
        df.to_parquet(path, index=False)
        saved[artifact_name] = path

    return saved


# ================================================================================================
# Build sorted cluster assignment dataframe with cluster sizes
# ================================================================================================
def build_cluster_list_df(
    item_keys: np.ndarray,
    labels: np.ndarray,
    key_col: str,
) -> pd.DataFrame:
    out = pd.DataFrame({
        key_col: item_keys.astype(int),
        "cluster_id": labels.astype(int),
    })
    out["cluster_size"] = out.groupby("cluster_id")["cluster_id"].transform("size").astype(int)
    return out.sort_values(["cluster_id", key_col]).reset_index(drop=True)


# ================================================================================================
# Resolve numeric feature columns for clustering from either persisted selection or fresh pruning
# ================================================================================================
def resolve_numeric_feature_cols(
    df: pd.DataFrame,
    key_col: str,
    selected_feature_cols: list[str] | None = None,
    corr_threshold: float = 0.7,
    empty_error_message: str = "No numeric feature columns found.",
) -> list[str]:
    if selected_feature_cols is None:
        feature_cols = remove_correlated_features(
            df=df,
            exclude_cols=[key_col],
            corr_threshold=corr_threshold,
        )
    else:
        feature_cols = [
            col
            for col in selected_feature_cols
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
        ]

    if not feature_cols:
        raise ValueError(empty_error_message)

    return feature_cols
