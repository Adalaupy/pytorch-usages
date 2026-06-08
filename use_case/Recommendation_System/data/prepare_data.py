from utils import kaggle_download
import pandas as pd
import re


# Project-wide schema constants.
ANIME_KEY_COL = 'anime_key'
USER_KEY_COL = 'user_key'
RATING_COL = 'rating'
TYPE_COL = 'type'
GENRE_COL = 'genre'
EPISODES_COL = 'episodes'
MEMBERS_COL = 'members'
ANIME_META_RATING_SOURCE_COL = 'rating'


def _sanitize_label(label):
    """Normalize label text into a safe lowercase token for column names."""
    value = re.sub(r'[^a-z0-9]+', '_', str(label).strip().lower()).strip('_')
    return value or 'unknown'


def _to_ref_cols(values, prefix):
    """Create full OHE reference column names from normalized values."""
    return [f'{prefix}_{value}' for value in values]


def _series_to_ohe(series, prefix, ref_cols=None):
    """Build one-hot columns and align to the provided reference schema."""
    ohe = pd.get_dummies(series, prefix=prefix, dtype=int)
    for col in ref_cols:
        if col not in ohe.columns:
            ohe[col] = 0
    ohe = ohe[ref_cols]
    return ohe


def _unique_type_values(df_anime, type_col='type'):
    """Extract unique normalized type labels from anime metadata."""
    return (
        df_anime[type_col]
        .fillna('unknown')
        .astype(str)
        .map(_sanitize_label)
        .drop_duplicates()
        .tolist()
    )


def _unique_genre_values(df_anime, genre_col='genre'):
    """Extract unique normalized genre labels from comma-separated genre strings."""
    return (
        df_anime[genre_col]
        .fillna('unknown')
        .astype(str)
        .str.split(r'\s*,\s*')
        .explode()
        .fillna('unknown')
        .map(_sanitize_label)
        .drop_duplicates()
        .tolist()
    )


def _load_and_index_csv(local_path, file_name):
    """Load one source CSV and add anime/user index keys when available."""
    df = pd.read_csv(f'{local_path}\\{file_name}.csv')
    df = Re_Indexing(df, 'anime_id', ANIME_KEY_COL)
    df = Re_Indexing(df, 'user_id', USER_KEY_COL)
    return df


# ================================================================================================
# Convert an original ID column into a contiguous integer key for model indexing.
# ================================================================================================

def Re_Indexing(df, unique_col, new_idx_col):

    if unique_col in df.columns:
        unique_id = df[unique_col].unique()
        idx_mapping = {original: idx for idx, original in enumerate(unique_id)}
        df[new_idx_col] = df[unique_col].map(idx_mapping)
    return df


# ================================================================================================
# Load anime and rating data from Kaggle, add index keys, and return both DataFrames.
# ================================================================================================

def get_data_indexing():

    # Download dataset and load the two source tables.
    local_path = kaggle_download('CooperUnion/anime-recommendations-database')
    df_anime = _load_and_index_csv(local_path, 'anime')
    df_rating = _load_and_index_csv(local_path, 'rating')

    # Build stable reference columns for aligned one-hot features.
    type_ref_cols = _to_ref_cols(_unique_type_values(df_anime), 'is_type')
    genre_ref_cols = _to_ref_cols(_unique_genre_values(df_anime), 'is_genre')
    return df_anime, df_rating, type_ref_cols, genre_ref_cols


def _compute_user_events(df_rating):

    events = df_rating[[USER_KEY_COL, ANIME_KEY_COL, RATING_COL]].copy()
    events = events.dropna(subset=[USER_KEY_COL, ANIME_KEY_COL])
    events[USER_KEY_COL] = events[USER_KEY_COL].astype(int)
    events[ANIME_KEY_COL] = events[ANIME_KEY_COL].astype(int)
    events[RATING_COL] = pd.to_numeric(events[RATING_COL], errors='coerce').astype(float)
    events['user_event_seq'] = events.groupby(USER_KEY_COL).cumcount() + 1

    cols = [
        USER_KEY_COL,
        ANIME_KEY_COL,
        RATING_COL,
        'user_event_seq',
    ]
    return events[cols].reset_index(drop=True)


def _compute_anime_content_ohe(df_anime, type_ref_cols, genre_ref_cols):

    content = df_anime[[ANIME_KEY_COL, TYPE_COL, GENRE_COL, EPISODES_COL, MEMBERS_COL, ANIME_META_RATING_SOURCE_COL]].copy()
    content[ANIME_KEY_COL] = content[ANIME_KEY_COL].astype(int)
    content['episodes_count'] = pd.to_numeric(content[EPISODES_COL], errors='coerce')
    content['members_count'] = pd.to_numeric(content[MEMBERS_COL], errors='coerce')
    content['anime_meta_rating'] = pd.to_numeric(content[ANIME_META_RATING_SOURCE_COL], errors='coerce')

    type_series = content[TYPE_COL].fillna('unknown').astype(str).map(_sanitize_label)
    type_ohe = _series_to_ohe(type_series, 'is_type', type_ref_cols)

    genre_series = content[GENRE_COL].fillna('unknown').astype(str)
    genre_exploded = genre_series.str.split(r'\s*,\s*').explode().fillna('unknown').map(_sanitize_label)
    genre_ohe = _series_to_ohe(genre_exploded, 'is_genre', genre_ref_cols)
    genre_ohe = genre_ohe.groupby(level=0).max()

    base_cols = [ANIME_KEY_COL, 'episodes_count', 'members_count', 'anime_meta_rating']
    return pd.concat([content[base_cols], type_ohe, genre_ohe], axis=1).reset_index(drop=True)


def _compute_user_profile_stats(df_user_event, df_anime_ohe, type_ref_cols, genre_ref_cols):

    anime_key_col = ANIME_KEY_COL
    user_key_col = USER_KEY_COL
    real_rating_mask = df_user_event[RATING_COL].notna() & df_user_event[RATING_COL].ne(-1)
    rated_events = df_user_event[real_rating_mask]

    user_base = df_user_event.groupby(user_key_col, as_index=False).agg(
        n_watches_total=('anime_key', 'size'),
    )
    user_rating_counts = rated_events.groupby(user_key_col, as_index=False).agg(
        n_ratings_total=(RATING_COL, 'size'),
    )
    user_base = user_base.merge(user_rating_counts, on=user_key_col, how='left')
    user_base['n_ratings_total'] = user_base['n_ratings_total'].fillna(0).astype(int)

    global_rating_mean = rated_events[RATING_COL].mean()
    user_rating_stats = rated_events.groupby(user_key_col, as_index=False).agg(
        user_avg_rating=(RATING_COL, 'mean'),
        user_std_rating=(RATING_COL, lambda x: x.std(ddof=0)),
    )
    user_rating_stats['user_bias_vs_global'] = user_rating_stats['user_avg_rating'] - global_rating_mean
    user_stats = user_base.merge(user_rating_stats, on=user_key_col, how='left')

    flag_cols = genre_ref_cols + type_ref_cols
    events_with_keys = df_user_event[[user_key_col, anime_key_col]]
    watched_with_content = events_with_keys.merge(
        df_anime_ohe[[anime_key_col] + flag_cols],
        on=anime_key_col,
        how='left',
    ).fillna(0)
    user_flag_counts = watched_with_content.groupby(user_key_col, as_index=False)[flag_cols].sum()
    rename_map = {
        col: col.replace('is_genre_', 'watch_cnt_genre_', 1).replace('is_type_', 'watch_cnt_type_', 1)
        for col in flag_cols
    }
    user_flag_counts = user_flag_counts.rename(columns=rename_map)
    user_stats = user_stats.merge(user_flag_counts, on=user_key_col, how='left')

    count_cols = [col for col in user_stats.columns if col.startswith('watch_cnt_')]
    user_stats[count_cols] = user_stats[count_cols].fillna(0)
    denom = user_stats['n_watches_total'].where(user_stats['n_watches_total'] > 0)
    prop_df = user_stats[count_cols].div(denom, axis=0).fillna(0.0)
    prop_df.columns = [col.replace('watch_cnt_', 'watch_prop_', 1) for col in count_cols]
    user_stats = pd.concat([user_stats, prop_df], axis=1)

    return user_stats


def _compute_anime_profile_stats(df_user_event, df_anime_ohe):

    anime_key_col = ANIME_KEY_COL
    real_rating_mask = df_user_event[RATING_COL].notna() & df_user_event[RATING_COL].ne(-1)
    rated_events = df_user_event[real_rating_mask]

    anime_base = df_user_event.groupby(anime_key_col, as_index=False).agg(
        n_watches_total=(anime_key_col, 'size'),
    )
    anime_rating_counts = rated_events.groupby(anime_key_col, as_index=False).agg(
        n_ratings_total=(RATING_COL, 'size'),
    )
    anime_base = anime_base.merge(anime_rating_counts, on=anime_key_col, how='left')
    anime_base['n_ratings_total'] = anime_base['n_ratings_total'].fillna(0).astype(int)

    anime_rating_stats = rated_events.groupby(anime_key_col, as_index=False).agg(
        rating_mean=(RATING_COL, 'mean'),
        rating_std=(RATING_COL, lambda x: x.std(ddof=0)),
        rating_median=(RATING_COL, 'median'),
    )
    anime_stats = anime_base.merge(anime_rating_stats, on=anime_key_col, how='left')
    anime_stats['rating_rate'] = anime_stats['n_ratings_total'] / anime_stats['n_watches_total']

    content_numeric = df_anime_ohe[[anime_key_col, 'episodes_count', 'members_count']]
    anime_stats = anime_stats.merge(content_numeric, on=anime_key_col, how='left')

    return anime_stats


def _compute_global_reference_stats(df_user_event, df_anime_ohe, type_ref_cols, genre_ref_cols):

    anime_key_col = ANIME_KEY_COL
    real_rating_mask = df_user_event[RATING_COL].notna() & df_user_event[RATING_COL].ne(-1)
    rated_events = df_user_event[real_rating_mask]

    n_watches_total = len(df_user_event)
    n_ratings_total = len(rated_events)
    stats = {
        'global_rating_mean': rated_events[RATING_COL].mean(),
        'global_rating_std': rated_events[RATING_COL].std(ddof=0),
        'global_rating_rate': (n_ratings_total / n_watches_total) if n_watches_total > 0 else 0.0,
    }

    flag_cols = genre_ref_cols + type_ref_cols
    rated_with_content = rated_events[[anime_key_col, RATING_COL]].merge(
        df_anime_ohe[[anime_key_col] + flag_cols],
        on=anime_key_col,
        how='left',
    ).fillna(0)

    for col in genre_ref_cols:
        suffix = col.replace('is_genre_', '', 1)
        stats[f'mean_rating_genre_{suffix}'] = rated_with_content.loc[
            rated_with_content[col] == 1,
            RATING_COL,
        ].mean()

    for col in type_ref_cols:
        suffix = col.replace('is_type_', '', 1)
        stats[f'mean_rating_type_{suffix}'] = rated_with_content.loc[
            rated_with_content[col] == 1,
            RATING_COL,
        ].mean()

    return pd.DataFrame([stats])


def _compute_anime_relative_stats(df_anime_profile_stats, df_anime_ohe, df_global_reference_stats, type_ref_cols, genre_ref_cols):

    anime_key_col = ANIME_KEY_COL
    rel = df_anime_profile_stats[[anime_key_col, 'rating_mean', 'n_ratings_total']].copy()
    rel = rel.merge(df_anime_ohe[[anime_key_col] + genre_ref_cols + type_ref_cols], on=anime_key_col, how='left').fillna(0)

    global_mean = df_global_reference_stats.at[0, 'global_rating_mean']
    global_std = df_global_reference_stats.at[0, 'global_rating_std']

    genre_sum = rel[genre_ref_cols].sum(axis=1)
    type_sum = rel[type_ref_cols].sum(axis=1)
    rel['primary_genre_col'] = rel[genre_ref_cols].idxmax(axis=1).where(genre_sum > 0)
    rel['primary_type_col'] = rel[type_ref_cols].idxmax(axis=1).where(type_sum > 0)

    rel['primary_genre_mean'] = rel['primary_genre_col'].map(
        lambda col: df_global_reference_stats.at[0, f"mean_rating_genre_{col.replace('is_genre_', '', 1)}"] if pd.notna(col) else None
    )
    rel['primary_type_mean'] = rel['primary_type_col'].map(
        lambda col: df_global_reference_stats.at[0, f"mean_rating_type_{col.replace('is_type_', '', 1)}"] if pd.notna(col) else None
    )

    rel['delta_vs_global_mean'] = rel['rating_mean'] - global_mean
    rel['ratio_vs_global_mean'] = rel['rating_mean'] / global_mean
    rel['delta_vs_primary_genre_mean'] = rel['rating_mean'] - rel['primary_genre_mean']
    rel['delta_vs_type_mean'] = rel['rating_mean'] - rel['primary_type_mean']

    if pd.notna(global_std) and global_std != 0:
        rel['zscore_vs_global'] = (rel['rating_mean'] - global_mean) / global_std
    else:
        rel['zscore_vs_global'] = 0.0

    rel['confidence_weight'] = rel['n_ratings_total'] / (rel['n_ratings_total'] + 50)

    return rel[
        [
            anime_key_col,
            'delta_vs_global_mean',
            'ratio_vs_global_mean',
            'delta_vs_primary_genre_mean',
            'delta_vs_type_mean',
            'zscore_vs_global',
            'confidence_weight',
        ]
    ]


# ================================================================================================
# Interim table: Build canonical interaction event table from indexed rating data.
# ================================================================================================

def user_events_df():

    _, df_rating, _, _ = get_data_indexing()
    return _compute_user_events(df_rating)


# ================================================================================================
# Interim table: Build anime content table with cleaned numeric fields and one-hot encoded labels.
# ================================================================================================

def anime_content_ohe_df():

    df_anime, _, type_ref_cols, genre_ref_cols = get_data_indexing()
    return _compute_anime_content_ohe(df_anime, type_ref_cols, genre_ref_cols)


# ================================================================================================
# Interim table: Build user profile stats from events and anime content features.
# ================================================================================================

def user_profile_stats_df():

    df_anime, df_rating, type_ref_cols, genre_ref_cols = get_data_indexing()
    df_user_event = _compute_user_events(df_rating)
    df_anime_ohe = _compute_anime_content_ohe(df_anime, type_ref_cols, genre_ref_cols)
    return _compute_user_profile_stats(df_user_event, df_anime_ohe, type_ref_cols, genre_ref_cols)


# ================================================================================================
# Interim table: Build anime profile stats from events and anime content features.
# ================================================================================================

def anime_profile_stats_df():
    df_anime, df_rating, type_ref_cols, genre_ref_cols = get_data_indexing()
    df_user_event = _compute_user_events(df_rating)
    df_anime_ohe = _compute_anime_content_ohe(df_anime, type_ref_cols, genre_ref_cols)
    return _compute_anime_profile_stats(df_user_event, df_anime_ohe)


# ================================================================================================
# Interim table: Build global reference stats from events and anime content features.
# ================================================================================================

def global_reference_stats_df():
    df_anime, df_rating, type_ref_cols, genre_ref_cols = get_data_indexing()
    df_user_event = _compute_user_events(df_rating)
    df_anime_ohe = _compute_anime_content_ohe(df_anime, type_ref_cols, genre_ref_cols)
    return _compute_global_reference_stats(df_user_event, df_anime_ohe, type_ref_cols, genre_ref_cols)


# ================================================================================================
# Interim table: Build anime relative stats versus global and category baselines.
# ================================================================================================

def anime_relative_stats_df():
    df_anime, df_rating, type_ref_cols, genre_ref_cols = get_data_indexing()
    df_user_event = _compute_user_events(df_rating)
    df_anime_ohe = _compute_anime_content_ohe(df_anime, type_ref_cols, genre_ref_cols)
    df_anime_profile_stats = _compute_anime_profile_stats(df_user_event, df_anime_ohe)
    df_global_reference_stats = _compute_global_reference_stats(df_user_event, df_anime_ohe, type_ref_cols, genre_ref_cols)
    return _compute_anime_relative_stats(df_anime_profile_stats, df_anime_ohe, df_global_reference_stats, type_ref_cols, genre_ref_cols)


# ================================================================================================
# Interim tables: Build all interim tables in one pass from fixed data source.
# ================================================================================================

def build_all_interim_tables():

    df_anime, df_rating, type_ref_cols, genre_ref_cols = get_data_indexing()
    df_user_event = _compute_user_events(df_rating)
    df_anime_ohe = _compute_anime_content_ohe(df_anime, type_ref_cols, genre_ref_cols)
    df_user_profile_stats = _compute_user_profile_stats(df_user_event, df_anime_ohe, type_ref_cols, genre_ref_cols)
    df_anime_profile_stats = _compute_anime_profile_stats(df_user_event, df_anime_ohe)
    df_global_reference_stats = _compute_global_reference_stats(df_user_event, df_anime_ohe, type_ref_cols, genre_ref_cols)
    df_anime_relative_stats = _compute_anime_relative_stats(df_anime_profile_stats, df_anime_ohe, df_global_reference_stats, type_ref_cols, genre_ref_cols)

    return {
        'df_anime': df_anime,
        'df_rating': df_rating,
        'type_ref_cols': type_ref_cols,
        'genre_ref_cols': genre_ref_cols,
        'df_user_event': df_user_event,
        'df_anime_ohe': df_anime_ohe,
        'df_user_profile_stats': df_user_profile_stats,
        'df_anime_profile_stats': df_anime_profile_stats,
        'df_global_reference_stats': df_global_reference_stats,
        'df_anime_relative_stats': df_anime_relative_stats,
    }


# ================================================================================================
# Testing
# ================================================================================================

if __name__ == '__main__':
    
    tables = build_all_interim_tables()
    df_anime = tables['df_anime']
    df_rating = tables['df_rating']
    df_user_event = tables['df_user_event']
    df_anime_ohe = tables['df_anime_ohe']
    df_user_profile_stats = tables['df_user_profile_stats']
    df_anime_profile_stats = tables['df_anime_profile_stats']
    df_global_reference_stats = tables['df_global_reference_stats']
    df_anime_relative_stats = tables['df_anime_relative_stats']

    # Preview outputs.
    print(df_anime.head(2))
    print(df_rating.head(2))
    print(df_user_event.head(2))
    print(df_anime_ohe.head(2))
    print(df_user_profile_stats.head(2))
    print(df_anime_profile_stats.head(2))
    print(df_global_reference_stats.head(1))
    print(df_anime_relative_stats.head(2))
