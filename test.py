import random
import pandas as pd
from multiprocessing import Pool, cpu_count
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pdb

def generate_prediction(df):
    # Split the data into features and target
    X = df.drop('class', axis=1)
    y = df['class']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=40)

    # Create and train the Random Forest model
    rf_model = LogisticRegression(random_state=40)
    rf_model.fit(X_train, y_train)

    # Make predictions with the Random Forest model
    rf_y_pred = rf_model.predict(X_test)
    # pdb.set_trace()

    # Evaluate the Random Forest model using AUC metric
    rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:,1])
    return rf_auc

# Function to run a single experiment
def run_experiment(args):
    seed, df = args


    random.seed(seed)  # Ensure reproducibility for each process
    search_df = pd.DataFrame(columns=['Positive', 'Negative', 'Hard', 'AUC', 'AUC_E', 'AUC_H'])

    df_local = df.copy()  # Create a local copy of the DataFrame for each process
    n_classes = len(df_local['class'].unique())

    for i in range(50):  # Inner loop
        classes_size = random.randint(3, n_classes)

        for j in range(50):
            pos_size = random.randint(1, classes_size - 2)
            pos_class = random.sample(list(df_local['class'].unique()), k=pos_size)

            neg_size = classes_size - pos_size
            neg_class = [x for x in df_local['class'].unique() if x not in pos_class]
            neg_class = random.sample(neg_class, k=neg_size)

            df_pos_neg = df_local.copy()
            df_pos_neg = df_pos_neg[df_pos_neg['class'].isin(pos_class + neg_class)]
            df_pos_neg['class'] = df_pos_neg['class'].apply(lambda x: 'P' if x in pos_class else 'N')

            auc = generate_prediction(df_pos_neg)

            for k in range(50):
                hard_size = random.randint(1, neg_size - 1)
                hard_class = random.sample(neg_class, k=hard_size)

                easy_size = neg_size - hard_size
                easy_class = [x for x in neg_class if x not in hard_class]
                easy_class = random.sample(easy_class, k=easy_size)

                df_pos_easy = df_local.copy()
                df_pos_easy = df_pos_easy[df_pos_easy['class'].isin(pos_class + easy_class)]
                df_pos_easy['class'] = df_pos_easy['class'].apply(lambda x: 'P' if x in pos_class else 'N')
                auc_easy = generate_prediction(df_pos_easy)

                df_pos_hard = df_local.copy()
                df_pos_hard = df_pos_hard[df_pos_hard['class'].isin(pos_class + hard_class)]
                df_pos_hard['class'] = df_pos_hard['class'].apply(lambda x: 'P' if x in pos_class else 'N')
                auc_hard = generate_prediction(df_pos_hard)

                result = {'Positive': pos_class, 'Negative': neg_class, 'Hard': hard_class,
                          'AUC': auc, 'AUC_E': auc_easy, 'AUC_H': auc_hard}
                search_df = pd.concat([search_df, pd.DataFrame([result])], ignore_index=True)

    return search_df

# Main function to parallelize the workload
if __name__ == '__main__':
    dataset = "datasets/Nursery.csv"
    def print_bad_lines(line):
        print(f"Bad line: {line}")

    df = pd.read_csv(dataset, on_bad_lines=print_bad_lines, engine='python')


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
    df = df[df['class'] != 'recommend']  # Remove rows where class is 'recommend'

    num_cores = cpu_count()  # Get the number of CPU cores
    seeds = [random.randint(0, 10000) for _ in range(num_cores)]  # Generate random seeds for each process
    args = [(seed, df) for seed in seeds]

    with Pool(num_cores) as pool:
        results = pool.map(run_experiment, args)  # Run experiments in parallel

    # Combine results from all processes
    final_search_df = pd.concat(results, ignore_index=True)
    print(final_search_df)