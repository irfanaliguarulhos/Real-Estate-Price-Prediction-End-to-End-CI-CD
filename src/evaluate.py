import os
import joblib
import numpy as np
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import lightgbm as lgb
import mlflow

def save_model(model, model_path: str, model_name: str):
    """Save model to disk"""
    try:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        print(f"Saving the model to: {model_path}")
        joblib.dump(model, model_path)
        print(f"Model {model_name} saved successfully to {model_path}")
    except Exception as e:
        raise IOError(f"Error saving model to '{model_path}'. Details: {e}")

def calculate_rmse(y_true, y_pred):
    """Calculate RMSE score"""
    mse = mean_squared_error(y_true, y_pred)
    return np.sqrt(mse)

def encode_categorical_columns(df: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """
    Encode categorical columns using binary encoding.
    Args:
        df (pd.DataFrame): Input dataframe with categorical columns.
        target_column (str): Name of the target column.
    Returns:
        pd.DataFrame: Dataframe with encoded categorical columns.
    """
    categorical_columns = df.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        print(f"Encoding categorical columns: {list(categorical_columns)}")
        for col in categorical_columns:
            df[col] = df[col].astype('category').cat.codes
    return df

def sanitize_feature_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace special characters in column names with underscores.
    Args:
        df (pd.DataFrame): Input dataframe.
    Returns:
        pd.DataFrame: Dataframe with sanitized column names.
    """
    df.columns = [col.replace(" ", "_").replace(",", "_").replace("-", "_") for col in df.columns]
    return df

def main(input_path: str, model_output: str):
    """
    Train multiple models on the dataset and save the best model.
    Args:
        input_path (str): Path to the processed dataset.
        model_output (str): Path to save the best model.
    """
    # Load and validate data
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Dataset loaded with shape: {df.shape}")

    # Feature engineering - use binary encoding
    df = encode_categorical_columns(df, 'PRICE')
    print(f"Dataset shape after encoding: {df.shape}")

    # Sanitize feature names
    df = sanitize_feature_names(df)
    print(f"Dataset column names after sanitization: {list(df.columns)}")

    # Validate target column
    if 'PRICE' not in df.columns:
        raise ValueError("Target column 'PRICE' not found in dataset")

    # Split data
    X = df.drop(columns=['PRICE'])
    y = df['PRICE']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # Define models with optimized parameters
    models = {
        'LinearRegression': (LinearRegression(), {}),
        'ElasticNet': (
            ElasticNet(random_state=42), 
            {'alpha': [0.1, 1.0], 'l1_ratio': [0.2, 0.8]}
        ),
        'RandomForest': (
            RandomForestRegressor(random_state=42), 
            {'n_estimators': [100], 'max_depth': [10]}
        ),
        'XGBoost': (
            xgb.XGBRegressor(random_state=42), 
            {'n_estimators': [100], 'learning_rate': [0.1]}
        ),
        'LightGBM': (
            lgb.LGBMRegressor(random_state=42), 
            {'n_estimators': [100], 'learning_rate': [0.1]}
        )
    }

    # Train and evaluate models
    best_model, best_score, best_name = None, np.inf, None
    
    with mlflow.start_run():
        for name, (model, params) in models.items():
            print(f"\nTraining {name} with parameters: {params}")
            grid = GridSearchCV(
                model, 
                params, 
                cv=3, 
                scoring='neg_mean_squared_error',
                n_jobs=2,  # Reduce parallelism
                verbose=0,
                error_score='raise'  # Raise errors for debugging
            )
            
            try:
                grid.fit(X_train, y_train)
                preds = grid.predict(X_test)
                rmse = calculate_rmse(y_test, preds)
                
                print(f"{name} RMSE: {rmse:.2f}")
                mlflow.log_metric(f"{name}_rmse", rmse)

                if rmse < best_score:
                    best_score = rmse
                    best_model = grid.best_estimator_
                    best_name = name
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue

        # Log best model
        if best_model is None:
            raise ValueError("No model was successfully trained")
            
        mlflow.log_param('best_model', best_name)
        mlflow.log_metric('best_rmse', best_score)
        print(f"\nBest model: {best_name} (RMSE: {best_score:.2f})")

        # Save the best model
        save_model(best_model, model_output, best_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train models on the processed dataset.")
    parser.add_argument(
        '--input',
        type=str,
        default='src/processed/NY-House-Features.csv',
        help='Path to the processed dataset'
    )
    parser.add_argument(
        '--model-output',
        type=str,
        default='models/best_model.pkl',
        help='Path to save the best model'
    )
    args = parser.parse_args()
    main(args.input, args.model_output)