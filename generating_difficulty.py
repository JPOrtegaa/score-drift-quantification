import pandas as pd
import numpy as np

from multiprocessing import Pool, cpu_count
import pdb

from scipy.stats import gaussian_kde
from tqdm import tqdm

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


def preprocessing_dataset(df):
    # Define the categories with the desired order
    categories = [
        ['usual', 'pretentious', 'great_pret'],  # parents
        ['proper', 'less_proper', 'improper', 'critical', 'very_crit'],  # has_nurs
        ['complete', 'completed', 'incomplete', 'foster'],  # form
        ['1', '2', '3', 'more'],  # children
        ['convenient', 'less_conv', 'critical'],  # housing
        ['convenient', 'inconv'],  # finance
        ['nonprob', 'slightly_prob', 'problematic'],  # social
        ['recommended', 'priority', 'not_recom']  # health
    ]

    # Initialize the OrdinalEncoder with the specified categories
    ordinal_encoder = OrdinalEncoder(categories=categories)

    classes = df.pop('class')
    columns = df.columns

    # Fit and transform the data
    df = ordinal_encoder.fit_transform(df)

    # Convert the result back to a DataFrame for better readability
    df = pd.DataFrame(df, columns=columns)
    df['class'] = classes

    return df

def compute_class_overlap(probs, labels, bandwidth='scott', num_points=1000):
    """
    Compute the overlap between the KDE-estimated probability distributions
    for the positive and negative classes.
    
    Parameters:
    - probs: array-like, predicted probabilities for the positive class (length = n_samples)
    - labels: array-like, true binary labels (0 or 1)
    - bandwidth: str or float, bandwidth method or value for KDE (default 'scott')
    - num_points: int, number of points for numerical integration (default 1000)

    Returns:
    - overlap: float, area under min(P_pos(x), P_neg(x)), ranges from 0 (no overlap) to 1 (complete)
    """
    probs = np.asarray(probs)
    labels = np.asarray(labels)

    probs_pos = probs[labels == 1]
    probs_neg = probs[labels == 0]

    if len(probs_pos) < 2 or len(probs_neg) < 2:
        raise ValueError("Need at least two samples per class for KDE.")

    kde_pos = gaussian_kde(probs_pos, bw_method=bandwidth)
    kde_neg = gaussian_kde(probs_neg, bw_method=bandwidth)

    x = np.linspace(0, 1, num_points)
    kde_vals_pos = kde_pos(x)
    kde_vals_neg = kde_neg(x)

    overlap = np.trapezoid(np.minimum(kde_vals_pos, kde_vals_neg), x)
    
    return overlap

# Function to generate predictions
def generate_prediction(df):
    # Split the data into features and target
    X = df.drop('class', axis=1)
    y = df['class']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=40)

    # Create and train the Logistic Regression model
    model = LogisticRegression(random_state=40)
    model.fit(X_train, y_train)

    # Evaluate the model using AUC metric
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    # Compute the class overlap
    overlap = compute_class_overlap(model.predict_proba(X_test)[:, 1], y_test)

    return auc, overlap

# Worker function to process a single row
# def process_row(row):
#     try:
#         # Extract the row data
#         pos_class = eval(row['Positive'])
#         neg_class = eval(row['Negative'])
#         easy_class = eval(row['Easy'])
#         hard_class = eval(row['Hard'])

#         # Create DataFrames for each scenario
#         df_pos_neg = df
#         df_pos_neg = df_pos_neg[df_pos_neg['class'].isin(pos_class + neg_class)]
#         df_pos_neg.loc[:, 'class'] = df_pos_neg['class'].apply(lambda x: 'P' if x in pos_class else 'N')
#         auc, overlap = generate_prediction(df_pos_neg)

#         df_pos_easy = df
#         df_pos_easy = df_pos_easy[df_pos_easy['class'].isin(pos_class + easy_class)]
#         df_pos_easy.loc[:, 'class'] = df_pos_easy['class'].apply(lambda x: 'P' if x in pos_class else 'N')
#         auc_easy, overlap_easy = generate_prediction(df_pos_easy)

#         df_pos_hard = df.copy()
#         df_pos_hard = df_pos_hard[df_pos_hard['class'].isin(pos_class + hard_class)]
#         df_pos_hard.loc[:, 'class'] = df_pos_hard['class'].apply(lambda x: 'P' if x in pos_class else 'N')
#         auc_hard, overlap_hard = generate_prediction(df_pos_hard)

#         # Return the results
#         return {
#             'Positive': pos_class,
#             'Negative': neg_class,
#             'Easy': easy_class,
#             'Hard': hard_class,
#             'AUC': auc,
#             'AUC_Easy': auc_easy,
#             'AUC_Hard': auc_hard,
#             'Overlap': overlap,
#             'Overlap_Easy': overlap_easy,
#             'Overlap_Hard': overlap_hard
#         }
#     except Exception as e:
#         print(f"Error processing row: {row}")
#         print(f"Exception: {e}")
#         return None

# Load the dataset
dataset_path = "datasets/Nursery.csv"
df = pd.read_csv(dataset_path)
df = preprocessing_dataset(df)

# pdb.set_trace()

def process_row(row):
    try:
        pos_class = eval(row['Positive'])
        neg_class = eval(row['Negative'])
        easy_class = eval(row['Easy'])
        hard_class = eval(row['Hard'])

        # Create DataFrames for each scenario
        df_pos_neg = df[df['class'].isin(pos_class + neg_class)].copy()
        df_pos_neg.loc[:, 'class'] = df_pos_neg['class'].apply(lambda x: 'P' if x in pos_class else 'N')
        auc, overlap = generate_prediction(df_pos_neg)

        df_pos_easy = df[df['class'].isin(pos_class + easy_class)].copy()
        df_pos_easy.loc[:, 'class'] = df_pos_easy['class'].apply(lambda x: 'P' if x in pos_class else 'N')
        auc_easy, overlap_easy = generate_prediction(df_pos_easy)

        df_pos_hard = df[df['class'].isin(pos_class + hard_class)].copy()
        df_pos_hard.loc[:, 'class'] = df_pos_hard['class'].apply(lambda x: 'P' if x in pos_class else 'N')
        auc_hard, overlap_hard = generate_prediction(df_pos_hard)

                # Return the results as a dictionary
        salve = {
            'Positive': pos_class,
            'Negative': neg_class,
            'Easy': easy_class,
            'Hard': hard_class,
            'AUC': auc,
            'AUC_Easy': auc_easy,
            'AUC_Hard': auc_hard,
            'Overlap': overlap,
            'Overlap_Easy': overlap_easy,
            'Overlap_Hard': overlap_hard
        }
        print(salve)

        # Return the results as a dictionary
        return {
            'Positive': pos_class,
            'Negative': neg_class,
            'Easy': easy_class,
            'Hard': hard_class,
            'AUC': auc,
            'AUC_Easy': auc_easy,
            'AUC_Hard': auc_hard,
            'Overlap': overlap,
            'Overlap_Easy': overlap_easy,
            'Overlap_Hard': overlap_hard
        }
    except Exception as e:
        print(f"Error processing row: {row}")
        print(f"Exception: {e}")
        return None

def running_esperiment_parallel(search_df):
    rows = [row for _, row in search_df.iterrows()]

    # Use multiprocessing to process rows in parallel
    num_cores = cpu_count()
    with Pool(num_cores) as pool:
        results = list(tqdm(pool.imap(process_row, rows), total=len(rows), desc="Running Experiment"))

    # Filter out None results (in case of errors)
    results = [res for res in results if res is not None]

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)
    return results_df

if __name__ == '__main__':
    # Load the search results
    search_results_path = "./search/search_results_nursery.csv"
    search_results = pd.read_csv(search_results_path)

    # Run the experiment in parallel
    results_df = running_esperiment_parallel(search_results)

    # Save the results to a new CSV file
    results_df.to_csv("processed_results_Covertype.csv", index=False)
    print("Processing complete. Results saved to 'processed_results_Covertype.csv'.")