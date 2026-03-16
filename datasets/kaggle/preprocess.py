import pandas as pd
import pdb
from sklearn.model_selection import train_test_split
from datasets.feature_selection import ensemble_feature_selection

def preprocess_cirrhosis(df):
    # Drop the ID column
    df = df.drop('ID', axis=1)
    
    # One-hot encoding for Status column
    status_dummies = pd.get_dummies(df['Status'], prefix='Status')
    df = pd.concat([df.drop('Status', axis=1), status_dummies], axis=1)
    
    # Drug column: transform to 0 and 1
    df['Drug'] = df['Drug'].map({'D-penicillamine': 0, 'Placebo': 1})
    
    # Sex: transform to 0 and 1
    df['Sex'] = df['Sex'].map({'M': 0, 'F': 1})
    
    # Ascites: transform to 1 and 0
    df['Ascites'] = df['Ascites'].map({'Y': 1, 'N': 0})
    
    # Hepatomegaly: transform to 0 and 1
    df['Hepatomegaly'] = df['Hepatomegaly'].map({'Y': 1, 'N': 0})
    
    # Spiders: transform to 1 and 0
    df['Spiders'] = df['Spiders'].map({'Y': 1, 'N': 0})
    
    # Edema: one-hot encoding
    edema_dummies = pd.get_dummies(df['Edema'], prefix='Edema')
    df = pd.concat([df.drop('Edema', axis=1), edema_dummies], axis=1)
    
    return df


def preprocess_predictive_maintenance(df):
    # Drop UDI and Product ID columns
    df = df.drop(['UDI', 'Product ID'], axis=1)
    
    # One-hot encoding for Type column
    type_dummies = pd.get_dummies(df['Type'], prefix='Type')
    df = pd.concat([df.drop('Type', axis=1), type_dummies], axis=1)
    
    # Drop Target column
    df = df.drop('Target', axis=1)
    
    return df


def preprocess_star_classification(df):
    df, _ = train_test_split(df, train_size=2000, stratify=df['class'], random_state=42)

    # Drop ID columns
    id_columns = ['obj_ID', 'run_ID', 'rerun_ID', 'cam_col', 'field_ID', 'spec_obj_ID']
    df = df.drop(id_columns, axis=1)
    
    # Drop plate, MJD, and fiber_ID
    df = df.drop(['plate', 'MJD', 'fiber_ID'], axis=1)
    
    return df


def preprocess_student_performance(df):
    # Drop GPA and StudentID columns
    df = df.drop(['GPA', 'StudentID'], axis=1)
    
    return df


def preprocess_zoo(df):
    # Drop animal_name column
    df = df.drop('animal_name', axis=1)
    
    return df


def preprocess_zoo2(df):
    df = df.rename(columns={'class_type': 'class'})
    return df


def preprocess_zoo3(df):
    df = df.rename(columns={'class_type': 'class'})
    return df


def preprocess_healthcare(df):
    df, _ = train_test_split(df, train_size=10000, stratify=df['class'], random_state=42)

    # Drop ID and code columns
    columns_to_drop = ['case_id', 'Hospital_code', 'Hospital_type_code', 'City_Code_Hospital', 
                       'Hospital_region_code', 'Ward_Type', 'Ward_Facility_Code', 
                       'patientid', 'City_Code_Patient']
    df = df.drop(columns_to_drop, axis=1)
    
    # One-hot encoding for Department
    department_dummies = pd.get_dummies(df['Department'], prefix='Department')
    df = pd.concat([df.drop('Department', axis=1), department_dummies], axis=1)
    
    # One-hot encoding for Type of Admission
    admission_dummies = pd.get_dummies(df['Type of Admission'], prefix='Type_of_Admission')
    df = pd.concat([df.drop('Type of Admission', axis=1), admission_dummies], axis=1)
    
    # Severity of Illness: transform to numbers (Minor=0, Moderate=1, Extreme=2)
    df['Severity of Illness'] = df['Severity of Illness'].map({'Minor': 0, 'Moderate': 1, 'Extreme': 2})
    
    # Age: convert range to midpoint
    def age_to_midpoint(age_str):
        if pd.isna(age_str):
            return None
        age_str = str(age_str)
        if '-' in age_str:
            lower, upper = age_str.split('-')
            return (int(lower) + int(upper)) / 2
        return float(age_str) if age_str.replace('.', '').isdigit() else None
    
    df['Age'] = df['Age'].apply(age_to_midpoint)
    
    return df


def preprocess_music_genre(df):
    # Drop Artist Name and Track Name columns
    df = df.drop(['Artist Name', 'Track Name'], axis=1)
    
    return df


def preprocess_fashion_mnist(df):
    df, _ = train_test_split(df, train_size=10000, stratify=df['class'], random_state=42)
    df = ensemble_feature_selection(df)
    return df


def preprocess_customer_segmentation(df):
    # Drop Var_1 column
    df = df.drop('Var_1', axis=1)
    
    # Spending_Score: transform to 0, 1, 2 (Low=0, Average=1, High=2)
    df['Spending_Score'] = df['Spending_Score'].map({'Low': 0, 'Average': 1, 'High': 2})
    
    # Gender: transform to 0 and 1
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
    
    # Ever_Married: transform to 0 and 1
    df['Ever_Married'] = df['Ever_Married'].map({'No': 0, 'Yes': 1})
    
    # Graduated: transform to 0 and 1
    df['Graduated'] = df['Graduated'].map({'No': 0, 'Yes': 1})
    
    # Profession: one-hot encoding
    profession_dummies = pd.get_dummies(df['Profession'], prefix='Profession')
    df = pd.concat([df.drop('Profession', axis=1), profession_dummies], axis=1)
    
    return df

if __name__ == "__main__":
    # Load the dataset
    df = pd.read_csv("customer_segmentation.csv")

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
    df = preprocess_customer_segmentation(df)
    print("\nData after preprocessing:")
    print(df.head())

    pdb.set_trace()

    # Save the preprocessed dataset
    # df.to_csv("datasets/kaggle/preprocessed_data.csv", index=False)
    # print("\nPreprocessed data saved to 'datasets/kaggle/preprocessed_data.csv'")