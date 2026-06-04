# Multi-Model Recommendation System Plan

## 1) Goal
Build one recommendation engine that runs multiple models for each user at the same time, then combines all model outputs into one final ranked anime list.

This design matches your idea: no single model is perfect for all users, so we use an ensemble.

---

## 2) High-Level Architecture
For each request (user U):

1. Run Model 1 to Model 5 in parallel.
2. Convert each model output into a comparable score scale.
3. Apply business-weighted blending.
4. Apply filtering/business rules.
5. Return Top-K recommendations.

Output = one final ranked list, with optional explainability tags (for example: "because similar users liked this").

---

## 3) Model-by-Model Design

### Model 1: User Clustering (Taste-Similar Users)
Your idea:
- Find users with similar tastes.
- Recommend from those similar users.

Recommended approach:
- Use matrix factorization style collaborative filtering (SVD or similar latent factors).
- Compute user-user similarity in latent space.
- Candidate source: anime watched/rated by top-N similar users.

Input:
- user_key
- watched percentage
- average rating by type/genre

Output:
- Candidate anime list from similar users
- Plus score per candidate (for blending)

Important filtering rules:
- Exclude items already watched by target user.
- Keep only high-quality candidates using real votes only (rating != -1), then apply rating threshold.
- Downweight globally over-popular items to reduce boring recommendations.

---

### Model 2: Anime Clustering (Content Similarity)
Your idea:
- Find similar anime based on each user's top liked anime and top genre.

Recommended approach:
- Content-based retrieval with vector similarity.
- Build anime vectors from genre, type, global rating, popularity, and optional text embeddings.
- Use cosine similarity or nearest-neighbor search.

Input:
- User's top rated anime history
- Anime feature vectors

Output:
- Similar anime candidates with similarity score

Best use case:
- Cold-start users (few ratings)
- New anime with limited interaction history

---

### Model 3: Predict User Rating for Anime
Your idea:
- Predict how much user U will rate anime A.

Recommended approach:
- Use your Simple feed-forward model.
- Features can include user embedding, anime embedding, and metadata features.
- Objective: regression score (predicted rating) or ordinal/binned rating class.

Input:
- User features + anime features

Output:
- Predicted rating score per (U, A)

Role in ensemble:
- Strong personalized relevance signal

---

### Model 4: Predict Next Anime (Sequence Model)
Your idea:
- Predict what user will watch next from watch sequence.

Recommended approach:
- Use your LSTM model as next-item predictor.
- Input sequence = user's interaction order from user_event_seq (pseudo-order, not real timestamp).
- Target = next anime_id.

Input:
- Ordered sequence of anime_key interactions by user_event_seq

Output:
- Probability distribution over anime catalog
- Top-N next-anime candidates

Role in ensemble:
- Captures short-term intent and session-like behavior

Current data caveat:
- user_event_seq is generated from source row order, so this is a proxy sequence model.
- Treat this model as optional/low-weight until real timestamps are available.

---

### Model 5: Predict Click / Not Click
Current practical version with existing data:
- Predict real-vote probability proxy for user U and anime A.
- Proxy label: 1 if rating != -1, else 0.

Recommended approach:
- Binary classifier is usually the simplest and strongest baseline (Logistic Regression, XGBoost, MLP).

Practical decision:
- Start with the proxy binary model first.
- Replace this with true click modeling later only if impression/click logs are available.

Input:
- User features, anime features (context features unavailable in current data)

Output:
- Proxy probability P(real_vote | U, A)

Role in ensemble:
- Useful as an engagement intent proxy with current data.

---

## 4) Candidate Generation + Ranking (Critical Missing Piece)
Do not score the full catalog with all heavy models each time.

Recommended two-stage pipeline:

1. Candidate Generation (fast)
- Use Model 1 + Model 2 + popularity/trending fallback
- Build candidate pool size, for example 300 to 1000 items

2. Candidate Scoring (heavier)
- Score candidates using Models 3, 4, 5
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

FinalScore(U, A) =
  w1 * S_user_cluster(U, A)
+ w2 * S_anime_cluster(U, A)
+ w3 * S_rating_pred(U, A)
+ w4 * S_next_item(U, A)
+ w5 * S_click_prob(U, A)

Constraints:
- w1 + w2 + w3 + w4 + w5 = 1
- All wi >= 0

Suggested starting weights (business prior):
- w1 = 0.20
- w2 = 0.15
- w3 = 0.25
- w4 = 0.20
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
- high_click_probability

---

## 9) Evaluation Plan (Offline Then Online)
Offline metrics:
- Precision@K
- Recall@K
- NDCG@K
- MAP@K
- RMSE or MAE for rating model only
- AUC/LogLoss for proxy real-vote model

Online metrics (A/B test):
- CTR
- Watch-through rate
- Completion rate
- Long-term retention
- Diversity and novelty metrics

---

## 10) Rollout Plan (No Data Processing Details Yet)
Phase A:
- Implement Model 1 + Model 2 + weighted combiner

Phase B:
- Add Model 3 (rating prediction)

Phase C:
- Add Model 4 (next-anime LSTM on pseudo sequence user_event_seq)

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

1. Generate candidates from Model 1 and Model 2.
2. Score each candidate with Models 3, 4, 5.
3. Normalize each model score to [0, 1].
4. Compute weighted final score.
5. Apply business filters and diversity rules.
6. Return Top-K anime recommendations.

This is the combined version of your idea plus production-ready missing pieces.
