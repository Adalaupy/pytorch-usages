# Interim Data Plan for Recommendation System

## Current Label Sets

<details>
<summary>Current type list (7)</summary>

- movie
- tv
- ova
- special
- music
- ona
- unknown

</details>

<details>
<summary>Current genre list (44)</summary>

- drama
- romance
- school
- supernatural
- action
- adventure
- fantasy
- magic
- military
- shounen
- comedy
- historical
- parody
- samurai
- sci_fi
- thriller
- sports
- super_power
- space
- slice_of_life
- mecha
- music
- mystery
- seinen
- martial_arts
- vampire
- shoujo
- horror
- police
- psychological
- demons
- ecchi
- josei
- shounen_ai
- game
- dementia
- harem
- cars
- kids
- shoujo_ai
- unknown
- hentai
- yaoi
- yuri

</details>

## 1) interaction_events
- Introduction: Canonical event-level table used as the single source for downstream feature tables.
- Granularity: One row per user-anime interaction event.

| Column | Type | Explanation | Sample |
|---|---|---|---|
| user_key | int | Re-indexed user ID for modeling | 102 |
| anime_key | int | Re-indexed anime ID for modeling | 445 |
| rating | float | Numeric rating from source, keeps -1 for non-vote watch records | 8.0 |
| user_event_seq | int | Per-user sequence generated from original row order | 17 |

Notes:
- This table should be built first and used to derive all other tables.
- There is no timestamp in source; sequence is a proxy for order.

---

## 2) anime_content_ohe
- Introduction: Static anime metadata and one-hot encoded content labels.
- Granularity: One row per anime.

| Column | Type | Explanation | Sample |
|---|---|---|---|
| anime_key | int | Re-indexed anime ID | 445 |
| episodes_count | float | Cleaned numeric episode count | 24 |
| members_count | float | Popularity proxy from members | 120430 |
| anime_meta_rating | float | Rating from anime metadata table (item-side metadata) | 8.42 |
| is_type_{type_name} | int (0/1) | Type one-hot pattern for each type label | is_type_tv |
| is_genre_{genre_name} | int (0/1) | Genre one-hot pattern for each genre label | is_genre_action |

Notes:
- Keep this table purely content/static, without user behavior aggregates.
- Current generated counts: 7 type columns and 44 genre columns.

---

## 3) user_profile_stats
- Introduction: User behavior and preference summary for personalization and bias handling.
- Granularity: One row per user.

| Column | Type | Explanation | Sample |
|---|---|---|---|
| user_key | int | Re-indexed user ID | 102 |
| n_watches_total | int | Total interaction records for the user | 315 |
| n_ratings_total | int | Count of real votes where rating != -1 | 204 |
| user_avg_rating | float | Mean of real-vote ratings | 7.65 |
| user_std_rating | float | Std dev of real-vote ratings | 1.24 |
| user_bias_vs_global | float | user_avg_rating - global_rating_mean | 0.38 |
| watch_cnt_genre_{genre_name} | int | Watched-count feature per genre | watch_cnt_genre_action |
| watch_prop_genre_{genre_name} | float | Genre watched proportion per user | watch_prop_genre_action |
| watch_cnt_type_{type_name} | int | Watched-count feature per type | watch_cnt_type_tv |
| watch_prop_type_{type_name} | float | Type watched proportion per user | watch_prop_type_tv |

Notes:
- Dynamic columns follow the same type/genre cardinality as anime_content_ohe.

---

## 4) anime_profile_stats
- Introduction: Anime-level rating and engagement statistics from interaction data.
- Granularity: One row per anime.

| Column | Type | Explanation | Sample |
|---|---|---|---|
| anime_key | int | Re-indexed anime ID | 445 |
| n_watches_total | int | Total watch count | 19488 |
| n_ratings_total | int | Total real-vote rating count (rating != -1) | 16210 |
| rating_mean | float | Mean real-vote rating | 7.67 |
| rating_std | float | Std dev real-vote rating | 1.31 |
| rating_median | float | Median real-vote rating | 8.0 |
| rating_rate | float | n_ratings_total / n_watches_total | 0.83 |
| episodes_count | float | Copied from content table | 24 |
| members_count | float | Copied from content table | 120430 |

Notes:
- No shrinkage is applied; this table keeps direct observed rating stats.

---

## 5) global_reference_stats
- Introduction: Global baseline statistics for normalization and relative comparisons.
- Granularity: One row for the whole dataset (wide format).

| Column | Type | Explanation | Sample |
|---|---|---|---|
| global_rating_mean | float | Mean real-vote rating overall | 7.27 |
| global_rating_std | float | Std dev of real-vote ratings | 1.49 |
| global_rating_rate | float | Real votes / total watch records | 0.79 |
| mean_rating_genre_{genre_name} | float | Global mean rating per genre | mean_rating_genre_action |
| mean_rating_type_{type_name} | float | Global mean rating per type | mean_rating_type_tv |

Notes:
- Can be persisted as a one-row table or serialized config.
- Dynamic columns follow the same type/genre cardinality as anime_content_ohe.

---

## 6) anime_relative_stats
- Introduction: Relative performance metrics comparing each anime against global/category baselines.
- Granularity: One row per anime.

| Column | Type | Explanation | Sample |
|---|---|---|---|
| anime_key | int | Re-indexed anime ID | 445 |
| delta_vs_global_mean | float | rating_mean - global_rating_mean | 0.40 |
| ratio_vs_global_mean | float | rating_mean / global_rating_mean | 1.06 |
| delta_vs_primary_genre_mean | float | rating_mean - primary_genre_mean | 0.22 |
| delta_vs_type_mean | float | rating_mean - primary_type_mean | 0.37 |
| zscore_vs_global | float | (rating_mean - global_mean) / global_std | 0.27 |
| confidence_weight | float | Reliability weight from sample size: n_ratings_total / (n_ratings_total + 50) | 0.95 |

Notes:
- Primary genre/type are selected from anime OHE flags.
- confidence_weight increases with more votes and stays below 1.

---

## Build Order
1. Build interaction_events.
2. Build anime_content_ohe.
3. Build user_profile_stats and anime_profile_stats.
4. Build global_reference_stats.
5. Build anime_relative_stats.

## Usage Guidance
- Model 1 (user clustering): user_profile_stats + interaction_events
- Model 2 (anime similarity): anime_content_ohe + anime_relative_stats
- Model 3 (rating prediction): interaction_events + user_profile_stats + anime_profile_stats + anime_content_ohe
- Model 4/5 (sequence/click): interaction_events (using user_event_seq as order feature)
