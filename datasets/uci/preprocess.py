import pandas as pd

import pdb

def preprocess_article_influence(df):

    # Drop rows where the class has 1 instance
    class_counts = df['class'].value_counts()
    classes_to_keep = class_counts[class_counts > 1].index
    df = df[df['class'].isin(classes_to_keep)]

    # Drop the issn column
    df = df.drop('issn', axis=1)
    
    return df