import argparse
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, r2_score
import os

def main(model_path: str, test_data: str):
    """
    Evaluate a trained model on test data.
    Args:
        model_path (str): Path to the trained model file.
        test_data (str): Path to the test dataset CSV file.
    """
    # Debugging: Print the paths
    print(f"Model path: {model_path}")
    print(f"Test data path: {test_data}")

    # Check if the model file exists
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"The model file '{model_path}' does not exist. Please check the path.")

    # Check if the test data file exists
    if not os.path.isfile(test_data):
        raise FileNotFoundError(f"The test data file '{test_data}' does not exist. Please check the path.")

    # Load the test dataset
    try:
        df = pd.read_csv(test_data)
        print(f"Test dataset loaded successfully with shape: {df.shape}")
    except Exception as e:
        raise ValueError(f"Error loading test dataset from {test_data}. Details: {e}")

    # Ensure the target column 'PRICE' exists (case-sensitive check)
    target_column = 'PRICE'  # Update this if your target column has a different name
    if target_column not in df.columns:
        raise ValueError(f"The target column '{target_column}' is missing from the test dataset.")

    # Split features and target
    X_test = df.drop(columns=[target_column])
    y_test = df[target_column]

    # Load the model
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
    except Exception as e:
        raise ValueError(f"Error loading model from {model_path}. Details: {e}")

    # Make predictions
    try:
        preds = model.predict(X_test)
        print("Predictions generated successfully.")
    except Exception as e:
        raise ValueError(f"Error generating predictions. Details: {e}")

    # Calculate metrics
    try:
        rmse = mean_squared_error(y_test, preds, squared=False)  # RMSE
        r2 = r2_score(y_test, preds)  # R² Score
        print(f"Evaluation Metrics:")
        print(f"RMSE: {rmse:.2f}")
        print(f"R2 Score: {r2:.2f}")
    except Exception as e:
        raise ValueError(f"Error calculating evaluation metrics. Details: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained model on test data.")
    parser.add_argument(
        '--model-path',
        type=str,
        default='/workspaces/Real-Estate-Price-Prediction-End-to-End-CI-CD/models/best_model.pkl',  # Default model path
        help='Path to the trained model file (default: /workspaces/Real-Estate-Price-Prediction-End-to-End-CI-CD/models/best_model.pkl)'
    )
    parser.add_argument(
        '--test-data',
        type=str,
        default='/workspaces/Real-Estate-Price-Prediction-End-to-End-CI-CD/src/processed/NY-House-Features.csv',  # Default dataset path
        help='Path to the test dataset CSV file (default: /workspaces/Real-Estate-Price-Prediction-End-to-End-CI-CD/src/processed/NY-House-Features.csv)'
    )
    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(args.model_path, args.test_data)