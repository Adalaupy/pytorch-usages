# User Preference Sorting Logic (Simple Early Version)

## 1) Scope
Early-stage logic only.
Use user interim table only: `interaction_events`.

Goal:
- For a target user, rank that user's favorite anime from past history.
- Return Top-K (default Top 10).

## 2) Signals (from interaction_events)
For each watched anime of user U:
- `user_watch_count`: how many times user U watched the anime.
- `user_avg_rating`: average of real ratings (rating != -1) by user U for that anime.
- `same_anime_watch_users`: how many unique users watched that anime globally.

## 3) Simple Score
Normalize each signal to [0, 1], then blend:

```text
score = 0.60 * rating_norm + 0.30 * watch_count_norm + 0.10 * same_watch_users_norm
```

Notes:
- If an anime has no real rating by this user, `rating_norm = 0`.
- If a signal has same value for all rows, normalized value is 0.

## 4) Sort Rule
Sort by:
1. `score` descending
2. `user_avg_rating` descending
3. `user_watch_count` descending
4. `anime_key` ascending

## 5) Output Columns
- `anime_key`
- `score`
- `user_avg_rating`
- `user_watch_count`
- `same_anime_watch_users`

## 6) Minimal Flow
1. Filter `interaction_events` by target `user_key`.
2. Aggregate per `anime_key`.
3. Join global `same_anime_watch_users`.
4. Compute score.
5. Sort and return Top 10.