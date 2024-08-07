import pandas as pd
import os
import logging
from src.data.data_preprocessing import load_data, preprocess_data, save_processed_data
from src.model.model_training import split_data, scale_data, train_logistic_regression, train_random_forest, evaluate_model, cross_validate_model
from src.visualization.visualization import plot_loan_status_distribution, plot_missing_values, plot_loan_amount_distribution, plot_confusion_matrix

logging.basicConfig(level=logging.INFO)

# Ensure the log file exists
log_file_exists = os.path.exists('app.log')
if not log_file_exists:
    with open('app.log', 'w') as f:
        f.write('Log file created.\n')

# Configure logging
def setup_logging():
    logging.basicConfig(level=logging.INFO, filename='app.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')

setup_logging()

def load_and_print_data(file_path):
    try:
        # Check if the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Load the dataset
        df = pd.read_csv(file_path)

        # Print the head of the DataFrame
        logging.info("DataFrame Head:")
        print(df.head())

        # Print the info of the DataFrame
        logging.info("\nDataFrame Info:")
        df.info()

        return df

    except FileNotFoundError as e:
        logging.error(f"{e}. Please ensure the file is in the correct directory.")
        logging.info("Current working directory: %s", os.getcwd())
        logging.info("Files in current directory: %s", os.listdir())
        return None

def main():
    file_path = 'credit.csv'
    df = load_and_print_data(file_path)
    
    if df is not None:
        logging.info("Visualizing loan status distribution...")
        plot_loan_status_distribution(df)
        
        logging.info("Visualizing missing values...")
        plot_missing_values(df)
        
        logging.info("Preprocessing data...")
        df = preprocess_data(df)
        save_processed_data(df, 'Processed_Credit_Dataset.csv')
        
        logging.info("Visualizing loan amount distribution...")
        plot_loan_amount_distribution(df)
        
        logging.info("Splitting data...")
        xtrain, xtest, ytrain, ytest = split_data(df)
        
        logging.info("Scaling data...")
        xtrain_scaled, xtest_scaled = scale_data(xtrain, xtest)
        
        logging.info("Training Logistic Regression model...")
        lrmodel = train_logistic_regression(xtrain_scaled, ytrain)
        lr_accuracy, lr_conf_matrix = evaluate_model(lrmodel, xtest_scaled, ytest)
        logging.info(f"Logistic Regression Accuracy: {lr_accuracy}")
        plot_confusion_matrix(lr_conf_matrix, title='Logistic Regression Confusion Matrix')
        
        logging.info("Training Random Forest model...")
        rfmodel = train_random_forest(xtrain_scaled, ytrain)
        rf_accuracy, rf_conf_matrix = evaluate_model(rfmodel, xtest_scaled, ytest)
        logging.info(f"Random Forest Accuracy: {rf_accuracy}")
        plot_confusion_matrix(rf_conf_matrix, title='Random Forest Confusion Matrix')
        
        logging.info("Cross-validating Logistic Regression model...")
        lr_mean, lr_std = cross_validate_model(lrmodel, xtrain_scaled, ytrain)
        logging.info(f"Logistic Regression Cross-Validation Mean Accuracy: {lr_mean}")
        logging.info(f"Logistic Regression Cross-Validation Std Dev: {lr_std}")
        
        logging.info("Cross-validating Random Forest model...")
        rf_mean, rf_std = cross_validate_model(rfmodel, xtrain_scaled, ytrain)
        logging.info(f"Random Forest Cross-Validation Mean Accuracy: {rf_mean}")
        logging.info(f"Random Forest Cross-Validation Std Dev: {rf_std}")
    else:
        logging.error("Unable to load data. Please check the file path and try again.")

if __name__ == "__main__":
    main()