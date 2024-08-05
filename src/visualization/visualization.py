import seaborn as sns
import matplotlib.pyplot as plt

def plot_loan_status_distribution(df):
    plt.figure(figsize=(10, 6))
    df['Loan_Status'].value_counts().plot.bar()
    plt.title('Loan Status Distribution')
    plt.xlabel('Loan Status')
    plt.ylabel('Count')
    plt.show()

def plot_missing_values(df):
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.show()

def plot_loan_amount_distribution(df):
    plt.figure(figsize=(10, 6))
    sns.histplot(df['LoanAmount'], kde=True)
    plt.title('Loan Amount Distribution')
    plt.xlabel('Loan Amount')
    plt.ylabel('Frequency')
    plt.show()

def plot_confusion_matrix(conf_matrix, title='Confusion Matrix'):
    plt.figure(figsize=(10, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()