import numpy as np
import pandas as pd
from pathlib import Path

from use_case.Recommendation_System.data.prepare_data import (
	user_profile_stats_df,
)


USER_KEY_COL = "user_key"
DEFAULT_PARALLEL_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "parallel_model_outputs"


# ================================================================================================
# Run a lightweight k-means clustering with numpy
# ================================================================================================
def _kmeans_numpy(
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
		# Assign step.
		dist_sq = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
		new_labels = np.argmin(dist_sq, axis=1)

		if np.array_equal(labels, new_labels):
			break
		labels = new_labels

		# Update step.
		for cluster_id in range(n_clusters):
			mask = labels == cluster_id
			if not np.any(mask):
				# Re-seed empty cluster with a random sample.
				centroids[cluster_id] = X[rng.integers(0, n_samples)]
				continue
			centroids[cluster_id] = X[mask].mean(axis=0)

	return labels


# ================================================================================================
# Build standardized user feature matrix from profile stats
# ================================================================================================
def build_user_feature_matrix() -> tuple[np.ndarray, np.ndarray, list[str]]:
	df_user_profile_stats = user_profile_stats_df()

	# Select numeric user features except user id.
	feature_cols = [
		col
		for col in df_user_profile_stats.columns
		if col != USER_KEY_COL and pd.api.types.is_numeric_dtype(df_user_profile_stats[col])
	]

	if not feature_cols:
		raise ValueError("No numeric feature columns found in user_profile_stats.")

	users = df_user_profile_stats[USER_KEY_COL].astype(int).to_numpy()
	X = df_user_profile_stats[feature_cols].fillna(0.0).to_numpy(dtype=float)

	# Standardize columns to keep high-scale features from dominating similarity.
	mean = X.mean(axis=0, keepdims=True)
	std = X.std(axis=0, keepdims=True)
	std = np.where(std == 0, 1.0, std)
	Xs = (X - mean) / std
	return users, Xs, feature_cols


# ================================================================================================
# Define dynamic cluster count from current user base size
# ================================================================================================
def _dynamic_cluster_count(n_users: int) -> int:
	# Equation: k = round(sqrt(N)), where N is current number of users.
	# This keeps average cluster size near sqrt(N), scaling automatically as N changes.
	return max(2, int(round(np.sqrt(max(1, n_users)))))


# ================================================================================================
# Build whole user-base clustering list
# ================================================================================================
def main_build_user_clustering_list(
	n_clusters: int | None = None,
	max_iter: int = 100,
	random_state: int = 42,
) -> pd.DataFrame:
	users, X, _ = build_user_feature_matrix()
	users = users.astype(int)
	cluster_count = n_clusters if n_clusters is not None else _dynamic_cluster_count(len(users))

	labels = _kmeans_numpy(
		X=X,
		n_clusters=cluster_count,
		max_iter=max_iter,
		random_state=random_state,
	)

	out = pd.DataFrame({
		USER_KEY_COL: users,
		"cluster_id": labels.astype(int),
	})
	out["cluster_size"] = out.groupby("cluster_id")["cluster_id"].transform("size").astype(int)
	out = out.sort_values(["cluster_id", USER_KEY_COL]).reset_index(drop=True)
	return out


# ================================================================================================
# Run full offline clustering pipeline and save parquet output
# ================================================================================================
def main_run_user_clustering_to_parquet(
	output_dir: str | Path | None = None,
	n_clusters: int | None = None,
	max_iter: int = 100,
	random_state: int = 42,
) -> Path:
	output_path = Path(output_dir) if output_dir else DEFAULT_PARALLEL_OUTPUT_DIR
	output_path.mkdir(parents=True, exist_ok=True)

	df_cluster_list = main_build_user_clustering_list(
		n_clusters=n_clusters,
		max_iter=max_iter,
		random_state=random_state,
	)

	cluster_path = output_path / "user_cluster_list.parquet"

	df_cluster_list.to_parquet(cluster_path, index=False)
	return cluster_path

if __name__ == "__main__":
	cluster_path = main_run_user_clustering_to_parquet()
	print(f"Saved cluster list: {cluster_path}")

