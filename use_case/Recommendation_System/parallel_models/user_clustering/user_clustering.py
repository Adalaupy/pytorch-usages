import numpy as np
import pandas as pd
from pathlib import Path

from use_case.Recommendation_System.data.prepare_data import (
	user_profile_stats_df,
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


USER_KEY_COL = "user_key"
DEFAULT_PARALLEL_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "parallel_model_outputs"
DEFAULT_CORR_THRESHOLD = 0.7


# ================================================================================================
# Build standardized user feature matrix from profile stats
# ================================================================================================
def build_user_feature_matrix(
	selected_feature_cols: list[str] | None = None,
	corr_threshold: float = DEFAULT_CORR_THRESHOLD,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
	df_user_profile_stats = user_profile_stats_df()
	feature_cols = resolve_numeric_feature_cols(
		df=df_user_profile_stats,
		key_col=USER_KEY_COL,
		selected_feature_cols=selected_feature_cols,
		corr_threshold=corr_threshold,
		empty_error_message="No numeric feature columns found in user_profile_stats.",
	)

	users = df_user_profile_stats[USER_KEY_COL].astype(int).to_numpy()
	X = df_user_profile_stats[feature_cols].fillna(0.0).to_numpy(dtype=float)
	Xs = standardize_matrix(X)
	return users, Xs, feature_cols


# ================================================================================================
# Build whole user-base clustering list
# ================================================================================================
def build_user_cluster(
	n_clusters: int | None = None,
	max_iter: int = 100,
	random_state: int = 42,
	corr_threshold: float = DEFAULT_CORR_THRESHOLD,
	feature_bundle: tuple[np.ndarray, np.ndarray, list[str]] | None = None,
) -> pd.DataFrame:
	users, X, _ = feature_bundle if feature_bundle is not None else build_user_feature_matrix(corr_threshold=corr_threshold)
	users = users.astype(int)
	cluster_count = n_clusters if n_clusters is not None else dynamic_cluster_count(len(users), min_clusters=2)

	labels = kmeans_numpy(
		X=X,
		n_clusters=cluster_count,
		max_iter=max_iter,
		random_state=random_state,
	)

	return build_cluster_list_df(
		item_keys=users,
		labels=labels,
		key_col=USER_KEY_COL,
	)


# ================================================================================================
# Run full offline clustering pipeline and save parquet output
# ================================================================================================
def main_run_user_clustering_to_parquet(
	output_dir: str | Path | None = None,
	n_clusters: int | None = None,
	max_iter: int = 100,
	random_state: int = 42,
	corr_threshold: float = DEFAULT_CORR_THRESHOLD,
) -> Path:
	output_path = Path(output_dir) if output_dir else DEFAULT_PARALLEL_OUTPUT_DIR
	output_path.mkdir(parents=True, exist_ok=True)

	selected_features_path = output_path / "user_cluster_selected_features.parquet"
	selected_feature_cols = load_selected_feature_names(selected_features_path)

	feature_bundle = build_user_feature_matrix(
		selected_feature_cols=selected_feature_cols,
		corr_threshold=corr_threshold,
	)
	_, _, final_feature_cols = feature_bundle
	df_cluster_list = build_user_cluster(
		n_clusters=n_clusters,
		max_iter=max_iter,
		random_state=random_state,
		corr_threshold=corr_threshold,
		feature_bundle=feature_bundle,
	)

	saved = save_parquet_artifacts(
		output_dir=output_path,
		artifacts={
			"user_cluster_list": df_cluster_list,
			"user_cluster_selected_features": pd.DataFrame({"feature_name": final_feature_cols}),
		},
	)
	return saved["user_cluster_list"]

if __name__ == "__main__":
	cluster_path = main_run_user_clustering_to_parquet()
	print(f"Saved cluster list: {cluster_path}")

