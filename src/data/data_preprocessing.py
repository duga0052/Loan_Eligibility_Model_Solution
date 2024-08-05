import pandas as pd

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Impute missing values
    df['Gender'] = df['Gender'].fillna('Male')
    df['Married'] = df['Married'].fillna(df['Married'].mode()[0])
    df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])
    df['Self_Employed'] = df['Self_Employed'].fillna(df['Self_Employed'].mode()[0])
    df['LoanAmount'] = df['LoanAmount'].fillna(df['LoanAmount'].median())
    df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
    df['Credit_History'] = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
    
    # Drop 'Loan_ID' variable
    df = df.drop('Loan_ID', axis=1)
    
    # Create dummy variables
    df = pd.get_dummies(df, columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area'])
    
    return df

def save_processed_data(df, file_path):
    df.to_csv(file_path, index=None)