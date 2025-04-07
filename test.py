import pandas as pd
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

import pdb

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
    return auc

# Worker function to process a single row
def process_row(row):

    # Extract the row data
    pos_class = eval(row['Positive'])
    neg_class = eval(row['Negative'])
    easy_class = eval(row['Easy'])
    hard_class = eval(row['Hard'])


    # df = df.copy()

    # Create DataFrames for each scenario
    df_pos_neg = df
    df_pos_neg = df_pos_neg[df_pos_neg['class'].isin(pos_class + neg_class)]
    df_pos_neg['class'] = df_pos_neg['class'].apply(lambda x: 'P' if x in pos_class else 'N')
    auc = generate_prediction(df_pos_neg)

    df_pos_easy = df
    df_pos_easy = df_pos_easy[df_pos_easy['class'].isin(pos_class + easy_class)]
    df_pos_easy['class'] = df_pos_easy['class'].apply(lambda x: 'P' if x in pos_class else 'N')
    auc_easy = generate_prediction(df_pos_easy)

    df_pos_hard = df.copy()
    df_pos_hard = df_pos_hard[df_pos_hard['class'].isin(pos_class + hard_class)]
    df_pos_hard['class'] = df_pos_hard['class'].apply(lambda x: 'P' if x in pos_class else 'N')
    auc_hard = generate_prediction(df_pos_hard)


    pdb.set_trace()
    # Return the results
    return {
        'Positive': pos_class,
        'Negative': neg_class,
        'Easy': easy_class,
        'Hard': hard_class,
        'AUC': auc,
        'AUC_Easy': auc_easy,
        'AUC_Hard': auc_hard
    }

# Load the dataset
dataset_path = "datasets/Nursery.csv"
df = pd.read_csv(dataset_path)
df = preprocessing_dataset(df)

if __name__ == '__main__':
    # Load the search results
    search_results_path = "search_results.csv"
    search_results = pd.read_csv(search_results_path)

    # pdb.set_trace()

    # Use multiprocessing to process rows in parallel
    num_cores = cpu_count()

    rows = [row for _, row in search_results.iterrows()]
    # pdb.set_trace()

    process_row(rows[0])

    with Pool(num_cores) as pool:
        # Wrap the iterrows with tqdm for progress tracking
        results = pool.map(process_row, tqdm([row for _, row in search_results.iterrows()][0], 
                                             total=len(search_results), 
                                             desc="Processing Rows"))

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Save the results to a new CSV file
    results_df.to_csv("processed_results.csv", index=False)
    print("Processing complete. Results saved to 'processed_results.csv'.")