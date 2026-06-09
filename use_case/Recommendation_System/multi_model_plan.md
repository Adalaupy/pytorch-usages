# Multi-Model Recommendation System Plan

## 1) Goal
Build one recommendation engine that runs multiple models for each user at the same time, then combines all model outputs into one final ranked anime list.

This design matches your idea: no single model is perfect for all users, so we use an ensemble.

---

## 2) High-Level Architecture
For each request (user U):

1. Load precomputed offline suggestions from Model 1 and Model 2.
2. Score the suggestion shortlist using online models for this request.
3. Convert each model output into a comparable score scale.
4. Apply business-weighted blending.
5. Apply filtering/business rules and return Top-K recommendations.

Output = one final ranked list, with optional explainability tags (for example: "because similar users liked this").

Runtime split (quick view):

| Mode | Models | Notes |
|---|---|---|
| Offline full-set batch | Model 1, Model 2 | Precompute reusable artifacts/suggestions |
| Online per-request per-user | Model 4 (+ additional online scorers) | Recompute for current user/request |

---

## 3) Model-by-Model Design

### Model 1: User Clustering (Taste-Similar Users)
Idea:
- Find users with similar tastes.
- Group users by profile behavior features, then retrieve same-cluster users.

Approach:
- Build user feature matrix from `user_profile_stats` numeric columns.
- Remove highly correlated features using shared feature selection (`corr_threshold`, default 0.7).
- Standardize features (z-score) so large-scale columns do not dominate.
- Run custom NumPy K-means clustering on full user base.
- Dynamic cluster count equation used by default: `k = round(sqrt(N_users))`, with minimum 2.
- Online similar-user retrieval = users in the same `cluster_id` as target user.

Input:
- user_profile_stats

Output:
- `user_cluster_list.parquet` (`user_key`, `cluster_id`, `cluster_size`)
- `user_cluster_selected_features.parquet` (persisted selected feature list for reruns)

Execution mode:
- Offline batch on full user base (precompute cluster artifacts).

Serving behavior (current code path):
- Similar users are selected from same cluster and returned by `user_key` order (Top-K).
- Anime recommendation from similar users is handled in shared module:
  - Exclude target user's watched anime.
  - Use loved-anime signal (`rating >= love_rating_threshold`, default 8.0).
  - Keep anime with minimum support users (`min_support_users`, default 2).
  - Score = `0.65 * support_norm + 0.25 * loved_rating_norm + 0.10 * watch_confidence_norm`.

---

### Model 2: Anime Clustering (Content Similarity)
Idea:
- Build anime clusters offline, then suggest anime from the same clusters as the user's favorite anime.

Approach:
- Build anime vectors from `anime_content_ohe` + `anime_relative_stats`.
- Add title-series features from anime names (franchise key + sequel index) for series continuity (e.g. Avatar 1/2/3).
- Remove highly correlated numeric columns (`corr_threshold`, default 0.7).
- Standardize vectors and run custom NumPy K-means.
- Save anime cluster assignments.
- Online retrieval: get user's favorite anime, find their clusters, and suggest unseen anime from those clusters.

Input (directly used inside `anime_clustering.py`):
- anime_content_ohe
- anime_relative_stats

Upstream dependency:
- interaction_events is not loaded directly in `anime_clustering.py`, but it is used to compute `anime_relative_stats` in data preparation.

Output:
- `anime_cluster_list.parquet`
- `anime_cluster_selected_features.parquet`
- Online suggestion table from Model 2 (`anime_key`, `model2_score`, `favorite_overlap_count`, `quality_prior`)

Execution mode:
- Offline batch on full item catalog for cluster artifacts + online per-request same-cluster suggestion.

Best use case:
- Cold-start users (few ratings)
- New anime with limited interaction history

Current implementation files:
- `parallel_models/anime_clustering/anime_clustering.py`
- `parallel_models/anime_clustering/get_anime_cluster_suggestions.py`
- `shared/title_language.py`

---

### Model 3: Predict User Rating for Anime
Idea:
- Predict how much user U will rate anime A.

Approach:
- Use your Simple feed-forward model.
- Features can include user embedding, anime embedding, and metadata features.
- Objective: regression score (predicted rating) or ordinal/binned rating class.

Input:
- interaction_events
- user_profile_stats
- anime_profile_stats
- anime_content_ohe

Output:
- Predicted rating score per (U, A)

Role in ensemble:
- Strong personalized relevance signal

---

### Model 4: Predict Next Anime (Sequence Model)
Idea:
- Predict what user will watch next from watch sequence.

Approach:
- Use your LSTM model as next-item predictor.
- Input sequence = user's interaction order from user_event_seq (pseudo-order, not real timestamp).
- Target = next anime_key.

Input:
- interaction_events

Output:
- Probability distribution over anime catalog
- Top-N next-anime suggestions

Role in ensemble:
- Captures short-term intent and session-like behavior

Current data caveat:
- user_event_seq is generated from source row order, so this is a proxy sequence model.
- Treat this model as optional/low-weight until real timestamps are available.

Execution mode:
- Online per-request per-user.

---

### Model 5: Predict Real-Vote Proxy
Current practical version with existing data:
- Predict real-vote probability proxy for user U and anime A.
- Proxy label: 1 if rating != -1, else 0.

Approach:
- Binary classifier is usually the simplest and strongest baseline (Logistic Regression, XGBoost, MLP).

Practical decision:
- Start with the proxy binary model first.
- Replace this with true click modeling later only if impression/click logs are available.

Input:
- interaction_events
- user_profile_stats
- anime_profile_stats
- anime_content_ohe

Output:
- Proxy probability P(real_vote | U, A)

Role in ensemble:
- Useful as an engagement intent proxy with current data.

---

## 4) Suggestion Shortlist + Ranking (Critical Missing Piece)
Do not score the full catalog with all heavy models each time.

Recommended two-stage pipeline:

1. Suggestion Shortlist Build (fast)
- Use precomputed outputs from offline Model 1 + Model 2 + popularity/trending fallback
- Build shortlist size, for example 300 to 1000 items

2. Suggestion Scoring (heavier)
- Score shortlist anime online per request (including Model 4 for current user)
- Blend all model scores with weights

This reduces latency and cost while preserving quality.

---

## 5) Score Normalization Before Weighting
Each model score has different scale (for example similarity, rating, probability).
Normalize first so weights are meaningful.

Common options:
- Min-max scaling to [0, 1]
- Z-score then sigmoid
- Rank-based normalization

Recommended default:
- Convert each model output to [0, 1] per request batch.

---

## 6) Weighting Strategy for Models 1-5
Final score for user U and anime A:

```text
FinalScore(U, A) =
  w1 * S_user_cluster(U, A)
+ w2 * S_anime_cluster(U, A)
+ w3 * S_rating_pred(U, A)
+ w4 * S_next_item(U, A)
+ w5 * S_real_vote_proxy(U, A)
```

Constraints:
- w1 + w2 + w3 + w4 + w5 = 1
- All wi >= 0

Suggested starting weights (business prior):
- w1 = 0.20
- w2 = 0.25
- w3 = 0.30
- w4 = 0.05
- w5 = 0.20

How to adjust weights in practice:
- New users: increase w2 (content) and popularity fallback.
- Power users with long history: increase w1 and w4.
- If business goal is engagement: increase w5.
- If goal is long-term satisfaction/quality: increase w3 and quality constraints.

---

## 7) Business Rules Layer (After Weighting)
Apply deterministic rules after model blending:

1. Remove already watched items.
2. Enforce safety/content constraints (if needed).
3. Add freshness quota (for example at least 20% recent titles).
4. Add diversity control (do not return only one genre).
5. Add exploration quota (for example 10% long-tail titles).

This prevents model-only output from becoming repetitive.

---

## 8) Final Return Format
For each user, return:

- Ranked anime_key list (Top-K)
- Final blended score
- Optional reason tag from strongest contributing model

Example reason tags:
- similar_users_like_this
- similar_to_your_favorites
- high_predicted_rating
- likely_next_watch
- high_real_vote_probability

---

## 9) Evaluation Plan (Offline Then Online)
Offline metrics:
- Precision@K
- Recall@K
- NDCG@K
- MAP@K
- RMSE or MAE for rating model only
- AUC/LogLoss for proxy real-vote model

Online metrics (A/B test, future when product logs are available):
- CTR
- Watch-through rate
- Completion rate
- Long-term retention
- Diversity and novelty metrics

---

## 10) Rollout Plan (No Data Processing Details Yet)
Phase A:
- Implement Model 1 + Model 2 as offline full-set batch jobs + weighted combiner

Phase B:
- Add Model 3 (rating prediction)

Phase C:
- Add Model 4 (next-anime LSTM on pseudo sequence user_event_seq) and run online per request/user

Phase D:
- Add Model 5 proxy model (real-vote binary), compare classifiers

Phase E:
- Tune weights and business rules using offline metrics, then A/B test

---

## 11) Key Risks and Mitigations
Risk: Popularity bias (always recommending mainstream titles)
- Mitigation: diversity + long-tail quota

Risk: Cold start for new users
- Mitigation: rely more on Model 2 + popularity/trending priors

Risk: Latency too high with many models
- Mitigation: two-stage retrieval then scoring

Risk: Score scale mismatch
- Mitigation: strict normalization before weighting

---

## 12) Minimal Pseudo-Flow

1. Generate suggestions from Model 1 and Model 2.
2. Score each suggestion with Models 3, 4, 5.
3. Normalize each model score to [0, 1].
4. Compute weighted final score.
5. Apply business filters and diversity rules.
6. Return Top-K anime recommendations.

This is the combined version of your idea plus production-ready missing pieces.

---

## 13) Implemented Speed Layer: Shared Processing Cached Parquets
To reduce online latency for the similar-user recommendation path, add a reusable shared-processing layer that precomputes user-anime signals once and reuses them for requests.

Design principle:
- Precompute heavy aggregations offline.
- Keep online path focused on lookup + lightweight scoring.

Saved parquet artifacts (single shared output folder):
- parallel_model_outputs/user_cluster_list.parquet
- parallel_model_outputs/user_cluster_selected_features.parquet
- parallel_model_outputs/anime_cluster_list.parquet
- parallel_model_outputs/anime_cluster_selected_features.parquet
- parallel_model_outputs/user_watch_history.parquet
- parallel_model_outputs/anime_watch_stats.parquet
- parallel_model_outputs/user_loved_anime.parquet

Implemented reusable modules and entry points:
- shared/anime_interest.py
  - prepare_user_anime_signal_parquets(...)
  - recommend_anime_from_similar_users(...)
- shared/title_language.py
  - derive_series_key(...)
  - build_title_series_matrix(...)
- parallel_models/anime_clustering/anime_clustering.py
  - main_run_anime_clustering_to_parquet(...)
- parallel_models/anime_clustering/get_anime_cluster_suggestions.py
  - main_get_anime_cluster_suggestions(...)
- parallel_models/user_clustering/get_similar_user.py
  - main_prepare_recommendation_signal_parquets(...)
  - main_recommend_anime_from_similar_user(...)

Online request flow (fast path):
1. Get top similar users from user_cluster_list.parquet.
2. Load precomputed shared signal parquets.
3. Score suggested anime from similar users using support + loved signal + watch confidence.
4. Return Top-K recommendations.

Why this is important:
- Reusable across multiple recommendation models under shared.
- Lower compute and faster response at serving time.
- Stable artifact contract for offline/online separation.

---

## 14) User Preference Sorting Logic
The full per-user preference ranking details are moved to a dedicated file for readability:

- anime_interest.md
