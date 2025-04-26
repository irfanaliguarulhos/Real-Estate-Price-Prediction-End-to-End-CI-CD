import argparse
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import mlflow
import os

def encode_categorical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns using one-hot encoding.
    Args:
        df (pd.DataFrame): Input dataframe with categorical columns.
    Returns:
        pd.DataFrame: Dataframe with encoded categorical columns.
    """
    categorical_columns = df.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        print(f"Encoding categorical columns: {list(categorical_columns)}")
        encoder = OneHotEncoder(sparse_output=False, drop='first')  # Updated for newer scikit-learn versions
        encoded_cols = pd.DataFrame(
            encoder.fit_transform(df[categorical_columns]),
            columns=encoder.get_feature_names_out(categorical_columns),
            index=df.index
        )
        df = pd.concat([df.drop(columns=categorical_columns), encoded_cols], axis=1)
    return df

def main(input_path: str, model_output: str):
    """
    Train multiple models on the dataset and save the best model.
    Args:
        input_path (str): Path to the processed dataset.
        model_output (str): Path to save the best model.
    """
    # Debugging: Print the input path
    print(f"Input path: {input_path}")

    # Check if the input file exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"The input file '{input_path}' does not exist. Please check the path.")

    # Load the processed dataset
    try:
        df = pd.read_csv(input_path)
        print(f"Dataset loaded successfully with shape: {df.shape}")
    except Exception as e:
        raise ValueError(f"Error loading dataset from {input_path}. Details: {e}")

    # Encode categorical columns
    df = encode_categorical_columns(df)

    # Check if the target column exists
    if 'PRICE' not in df.columns:
        raise ValueError("The target column 'PRICE' is missing from the dataset.")

    # Split features and target
    X = df.drop(columns=['PRICE'])  # Ensure 'PRICE' matches the column name in your dataset
    y = df['PRICE']

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape}, Testing set size: {X_test.shape}")

    # Define models and hyperparameters
    models = {
        'LinearRegression': (LinearRegression(), {}),
        'ElasticNet': (ElasticNet(), {'alpha': [0.1, 1.0, 10.0]}),
        'RandomForest': (RandomForestRegressor(random_state=42), {'n_estimators': [100, 200], 'max_depth': [10, 20]}),
        'XGBoost': (xgb.XGBRegressor(random_state=42, objective='reg:squarederror'), {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]}),
        'LightGBM': (lgb.LGBMRegressor(random_state=42), {'n_estimators': [100, 200], 'learning_rate': [0.01, 0.1]})
    }

    # Start MLflow run
    mlflow.start_run()
    best_model, best_score, best_name = None, np.inf, None

    # Train and evaluate models
    for name, (model, params) in models.items():
        print(f"Training and evaluating model: {name}")
        grid = GridSearchCV(model, params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
        grid.fit(X_train, y_train)
        preds = grid.predict(X_test)
        rmse = mean_squared_error(y_test, preds, squared=False)
        mlflow.log_metric(f"{name}_rmse", rmse)
        print(f"{name} RMSE: {rmse}")

        # Track the best model
        if rmse < best_score:
            best_score = rmse
            best_model = grid.best_estimator_
            best_name = name

    # Log the best model name and score
    mlflow.log_param('best_model', best_name)
    mlflow.log_metric('best_rmse', best_score)
    mlflow.end_run()

    # Save the best model
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(model_output), exist_ok=True)
        print(f"Saving the best model to: {model_output}")
        
        # Save the model
        joblib.dump(best_model, model_output)
        print(f"Best model ({best_name}) saved to {model_output}")
    except Exception as e:
        raise IOError(f"Error saving the best model to {model_output}. Details: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train models on the processed dataset.")
    parser.add_argument(
        '--input',
        type=str,
        default='src/processed/NY-House-Features.csv',  # Default path to the processed dataset
        help='Path to the processed dataset (default: src/processed/NY-House-Features.csv)'
    )
    parser.add_argument(
        '--model-output',
        type=str,
        default='models/best_model.pkl',  # Default path to save the best model
        help='Path to save the best model (default: models/best_model.pkl)'
    )
    args = parser.parse_args()
    main(args.input, args.model_output)


