import random, pdb
from tqdm import tqdm

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
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Script for generating groups.")
parser.add_argument("-d", "--dataset", type=str, required=True, help="File name for the dataset")
args = parser.parse_args()

dataset_name = args.dataset
dataset = "./datasets/" + dataset_name + ".csv"

def print_bad_lines(line):
    print(f"Bad line: {line}")

df = pd.read_csv(dataset, on_bad_lines=print_bad_lines, engine='python')

# random.seed(45)
search_df = pd.DataFrame(columns=['Positive', 'Negative', 'Easy', 'Hard'])
search_df.to_csv("search_results_" + dataset_name + ".csv", index=False)

df.rename(columns={'Class': 'class'}, inplace=True)

n_classes = len(df['class'].unique())

for i in tqdm(range(50), desc="Total Classes"):

    classes_size = random.randint(3, n_classes)

    for j in tqdm(range(50), desc="Positive and Negative", leave=False):
        pos_size = random.randint(1, classes_size-2)
        pos_class = random.sample(list(df['class'].unique()), k=pos_size)

        neg_size = classes_size - pos_size
        neg_class = [x for x in df['class'].unique() if x not in pos_class]
        neg_class = random.sample(neg_class, k=neg_size)
        # print(classes_size, pos_class, neg_class)

        df_pos_neg = df.copy()
        df_pos_neg = df_pos_neg[df_pos_neg['class'].isin(pos_class + neg_class)]
        df_pos_neg['class'] = df_pos_neg['class'].apply(lambda x: 'P' if x in pos_class else 'N')

        for k in tqdm(range(50), desc="Easy and Hard", leave=False):
            hard_size = random.randint(1, neg_size-1)
            hard_class = random.sample(neg_class, k=hard_size)

            easy_size = neg_size - hard_size
            easy_class = [x for x in neg_class if x not in hard_class]
            easy_class = random.sample(easy_class, k=easy_size)

            df_pos_easy = df.copy()
            df_pos_easy = df_pos_easy[df_pos_easy['class'].isin(pos_class + easy_class)]
            df_pos_easy['class'] = df_pos_easy['class'].apply(lambda x: 'P' if x in pos_class else 'N')

            df_pos_hard = df.copy()
            df_pos_hard = df_pos_hard[df_pos_hard['class'].isin(pos_class + hard_class)]
            df_pos_hard['class'] = df_pos_hard['class'].apply(lambda x: 'P' if x in pos_class else 'N')

            result = {'Positive': pos_class, 'Negative': neg_class,
                      'Easy': easy_class, 'Hard': hard_class}
            search_df = pd.concat([search_df, pd.DataFrame([result])], ignore_index=True)

    search_df.to_csv("search_results_" + dataset_name + ".csv", index=False)