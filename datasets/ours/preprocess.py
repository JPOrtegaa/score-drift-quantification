import pandas as pd
from sklearn.model_selection import train_test_split
from datasets.feature_selection import ensemble_feature_selection


def preprocess_har(df):
    df = df.rename(columns={'Class': 'class'})
    df = ensemble_feature_selection(df)
    return df


def preprocess_covertype(df):
    df, _ = train_test_split(df, train_size=10000, stratify=df['class'], random_state=42)
    return df


def preprocess_dermatology(df):
    df, _ = train_test_split(df, train_size=10000, stratify=df['class'], random_state=42)
    return df


def preprocess_mosquitoes(df):
    df = df.drop(columns=['sensor_id', 'file', 'time_elapsed'], errors='ignore')
    df, _ = train_test_split(df, train_size=10000, stratify=df['class'], random_state=42)
    return df
