"""
Preprocessing for the Schumacher quantification-benchmark datasets.

Ported from Tobias Schumacher's paper repository
    https://github.com/tobiasschumacher/quantification_paper
which holds one `data/<name>/prep.py` per dataset. In that repo each
`prep_data()`:
  1. downloads the raw source,
  2. drops id / leakage columns and one-hot encodes categoricals,
  3. discretizes the (regression / continuous) target into the paper's classes
     via ``pd.cut`` — the label column per dataset is the ``target`` field in
     ``data/data_index.csv`` (e.g. bike->cnt, concrete->strength, drugs->att28),
  4. optionally (``binned=True``) quantile-bins the continuous features into 4
     bins with ``pd.qcut``.

The CSVs under ``datasets/schumacher/`` are already the ``prep_data(binned=False)``
output (steps 1-3 applied: target discretized, categoricals one-hot, ids dropped).
So the functions below take that on-disk frame and apply the remaining
``binned=True`` feature discretization on the same columns the paper uses, then
rename the label column to ``class`` to match the convention used across
``datasets/`` (cf. ``datasets/ours/preprocess.py``).

Label column per dataset (from data_index.csv):
    bike->cnt  blog_feedback->att280  concrete->strength  contraceptive->Contraceptive
    diamonds->cut  drugs->att28  energy->Appliances  fifa19->Wage  news_popularity->shares
    skillcraft->LeagueIndex  superconductor->critical_temp  turk_student_eval->instr
    video_game_sales->Critic_Score  yeast->class  theorem->res (constructed in prep)
"""

import pandas as pd


def _qbin(df, cols, q=4):
    """Schumacher's binned=True step: quantile-bin the given continuous columns
    into ``q`` bins (skips non-numeric / constant columns for safety)."""
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col]
        if not pd.api.types.is_numeric_dtype(s) or s.nunique(dropna=True) <= 1:
            continue
        df[col] = pd.qcut(s, q=q, labels=False, duplicates='drop').astype('int64')
    return df


def _relabel(df, target):
    """Rename the dataset's label column to the shared 'class' convention."""
    return df.rename(columns={target: 'class'}) if target in df.columns else df


def preprocess_bike(df):
    df = _qbin(df, list(df.columns)[2:6])  # temp, atemp, hum, windspeed
    return _relabel(df, 'cnt')


def preprocess_blog_feedback(df):
    df = _qbin(df, list(df.columns)[:59])
    df = _qbin(df, list(df.columns)[272:-1])
    return _relabel(df, 'att280')


def preprocess_concrete(df):
    df = _qbin(df, list(df.columns)[:-1])  # comp1..comp8
    return _relabel(df, 'strength')


def preprocess_contraceptive(df):
    # paper cut-bins Age and NumberChildren with fixed edges (not qcut)
    df = df.copy()
    df['Age'] = pd.cut(df['Age'], bins=[15, 25, 30, 35, 40, 50],
                       labels=[1, 2, 3, 4, 5]).astype('int64')
    df['NumberChildren'] = pd.cut(df['NumberChildren'], bins=[-1, 0, 1, 2, 3, 5, 20],
                                  labels=[0, 1, 2, 3, 4, 5]).astype('int64')
    return _relabel(df, 'Contraceptive')


def preprocess_diamonds(df):
    df = _qbin(df, ['carat', 'depth', 'table', 'price', 'xc', 'yc', 'zc'])
    return _relabel(df, 'cut')


def preprocess_drugs(df):
    df = _qbin(df, list(df.columns)[:12])  # att1..att12 (personality scores)
    return _relabel(df, 'att28')


def preprocess_energy(df):
    df = _qbin(df, list(df.columns)[1:])  # every column after Appliances
    return _relabel(df, 'Appliances')


def preprocess_fifa19(df):
    cols = [c for c in list(df.columns)[:72] if c != 'Wage']
    df = _qbin(df, cols)
    return _relabel(df, 'Wage')


def preprocess_news_popularity(df):
    qcols = [
        'timedelta', 'n_tokens_title', 'n_tokens_content', 'n_unique_tokens',
        'n_non_stop_words', 'n_non_stop_unique_tokens', 'num_hrefs', 'num_self_hrefs',
        'num_imgs', 'num_videos', 'average_token_length', 'num_keywords',
        'kw_min_min', 'kw_max_min', 'kw_avg_min', 'kw_min_max', 'kw_max_max',
        'kw_avg_max', 'kw_min_avg', 'kw_max_avg', 'kw_avg_avg',
        'self_reference_min_shares', 'self_reference_max_shares',
        'self_reference_avg_sharess', 'LDA_00', 'LDA_01', 'LDA_02', 'LDA_03',
        'LDA_04', 'global_subjectivity', 'global_sentiment_polarity',
        'global_rate_positive_words', 'global_rate_negative_words',
        'rate_positive_words', 'rate_negative_words', 'avg_positive_polarity',
        'min_positive_polarity', 'max_positive_polarity', 'avg_negative_polarity',
        'min_negative_polarity', 'max_negative_polarity', 'title_subjectivity',
        'title_sentiment_polarity', 'abs_title_subjectivity',
        'abs_title_sentiment_polarity',
    ]
    df = _qbin(df, qcols)
    return _relabel(df, 'shares')


def preprocess_skillcraft(df):
    df = _qbin(df, list(df.columns)[1:])  # every column after LeagueIndex
    return _relabel(df, 'LeagueIndex')


def preprocess_superconductor(df):
    df = _qbin(df, list(df.columns)[:-10])  # continuous features before the dummies
    return _relabel(df, 'critical_temp')


def preprocess_turk_student_eval(df):
    # the paper's turk prep has no binned mode — Likert features are kept as-is
    return _relabel(df, 'instr')


def preprocess_video_game_sales(df):
    df = _qbin(df, ['Year_of_Release', 'NA_Sales', 'EU_Sales', 'JP_Sales',
                    'Other_Sales', 'Critic_Count', 'User_Count'])
    return _relabel(df, 'Critic_Score')


def preprocess_yeast(df):
    # paper cut-bins these six features with fixed edges (not qcut)
    df = df.copy()
    for col in ['mcg', 'gvh', 'alm']:
        df[col] = pd.cut(df[col], bins=[0, 0.4, 0.5, 0.6, 1],
                         labels=[1, 2, 3, 4]).astype('int64')
    df['mit'] = pd.cut(df['mit'], bins=[-0.1, 0.1, 0.2, 0.3, 1],
                       labels=[1, 2, 3, 4]).astype('int64')
    df['pox'] = pd.cut(df['pox'], bins=[-0.1, 0.4, 0.5, 0.6, 1],
                       labels=[1, 2, 3, 4]).astype('int64')
    df['vac'] = pd.cut(df['vac'], bins=[-0.1, 0.25, 0.35, 1],
                       labels=[1, 2, 3]).astype('int64')
    return _relabel(df, 'class')


def preprocess_theorem(df):
    """First-order Theorem Proving. Unlike the other schumacher datasets, the
    on-disk file (first-order-theorem/all-data-raw.csv) is the *raw* prover-times
    matrix (58 unnamed columns), so this reproduces the full data/theorem/prep.py:
    the target `res` is which of the last five provers (att54..att58) solves the
    problem fastest, with a 'no prover within 100' class; then id/time columns are
    dropped, class 2 removed and class 5 folded into 2 -> 5 classes, 51 features.
    """
    df = df.copy()
    df.columns = ['att' + str(i + 1) for i in range(df.shape[1])]
    df[df == -100] = 1000
    df['min_time'] = df.iloc[:, -5:].min(axis=1)
    df['res'] = df.iloc[:, -6:-1].idxmin(axis=1)
    df['res'] = df['res'].apply(lambda t: int(t[-1]) - 3)
    df.loc[df['min_time'] > 100, 'res'] = 0
    df = df.drop(['att54', 'att55', 'att56', 'att57', 'att58', 'min_time', 'att5', 'att35'], axis=1)
    df = df[df['res'] != 2]
    df.loc[df['res'] == 5, 'res'] = 2
    df = _qbin(df, [c for c in df.columns if c != 'res'])
    return _relabel(df, 'res')
