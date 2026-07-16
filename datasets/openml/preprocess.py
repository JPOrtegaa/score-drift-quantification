import pandas as pd
import pdb
from sklearn.model_selection import train_test_split
from datasets.feature_selection import ensemble_feature_selection

def preprocess_spectrometer(df):
    # Drop ID-type column
    df = df.drop('ID-type', axis=1)

    # Keep only classes with at least 10 samples (as in the original ovr_results runs)
    counts = df['class'].value_counts()
    df = df[df['class'].isin(counts[counts >= 10].index)]

    return df


def preprocess_bach_choral_harmony(df):
    # Drop Choral ID (V1) and Event number (V2) columns
    df = df.drop(['V1', 'V2'], axis=1)
    
    # Convert YES/NO columns (V3-V14) to 0 and 1
    yes_no_columns = ['V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12', 'V13', 'V14']
    for col in yes_no_columns:
        df[col] = df[col].str.strip().map({'YES': 1, 'NO': 0})
    
    # V15 (Bass pitch class): one-hot encoding
    v15_dummies = pd.get_dummies(df['V15'], prefix='Bass')
    df = pd.concat([df.drop('V15', axis=1), v15_dummies], axis=1)

    # Keep only classes with at least 10 samples (as in the original ovr_results runs)
    counts = df['class'].value_counts()
    df = df[df['class'].isin(counts[counts >= 10].index)]

    return df


def preprocess_fars(df):
    # Keep only classes with at least 10 samples (as in the original ovr_results runs)
    counts = df['class'].value_counts()
    df = df[df['class'].isin(counts[counts >= 10].index)]

    df, _ = train_test_split(df, train_size=10000, stratify=df['class'], random_state=42)
    return df


def preprocess_amazon_commerce_reviews(df):
    df = ensemble_feature_selection(df)
    return df


def preprocess_fabert(df):
    df = ensemble_feature_selection(df)
    return df


def preprocess_amazon_commerce_reviews_subset(df):
    df = ensemble_feature_selection(df)
    return df


if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv("dataset_4552_BachChoralHarmony.csv")

    # Display the first few rows of the dataset
    print("First few rows of the dataset:")
    print(df.head())

    # Check for missing values
    missing_values = df.isnull().sum()
    print("\nMissing values in each column:")
    print(missing_values)

    # # Fill missing values with the mean of the column
    # df.fillna(df.mean(), inplace=True)

    # Verify that there are no more missing values
    missing_values_after = df.isnull().sum()
    print("\nMissing values after imputation:")
    print(missing_values_after)

    # Preprocess the student performance dataset
    df = preprocess_bach_choral_harmony(df)
    print("\nData after preprocessing:")
    print(df.head())

    pdb.set_trace()

    # Save the preprocessed dataset
    # df.to_csv("datasets/kaggle/preprocessed_data.csv", index=False)
    # print("\nPreprocessed data saved to 'datasets/kaggle/preprocessed_data.csv'")