import argparse
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import os

def main(input_path: str, output_path: str):
    """
    Perform feature engineering on the dataset and save the processed data.
    Args:
        input_path (str): Path to the cleaned CSV file.
        output_path (str): Path to save the processed dataset.
    """
    # Check if the input file exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"The input file does not exist: {input_path}")

    # Read the cleaned dataset
    df = pd.read_csv(input_path)

    # Check required columns exist in the dataset
    required_columns = ['BEDS', 'BATH', 'PROPERTYSQFT', 'STATE']  # Updated column names to match your dataset
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"The following required columns are missing in the dataset: {missing_columns}")

    # Impute missing numerical values
    imputer = SimpleImputer(strategy='median')
    num_cols = ['BEDS', 'BATH', 'PROPERTYSQFT']
    df[num_cols] = imputer.fit_transform(df[num_cols])

    # Encode categorical 'STATE'
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # Updated for newer scikit-learn versions
    cat_encoded = encoder.fit_transform(df[['STATE']])  # Ensure input is a DataFrame
    cat_cols = encoder.get_feature_names_out(['STATE'])
    df[cat_cols] = cat_encoded
    df.drop(columns=['STATE'], inplace=True)

    # Scale numerical features
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    # Persist transformers
    os.makedirs('models', exist_ok=True)
    joblib.dump(imputer, 'models/imputer.pkl')
    joblib.dump(encoder, 'models/encoder.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    # Save engineered features
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Feature-engineered dataset saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Feature engineering script for real estate data.")
    parser.add_argument(
        '--input', 
        type=str, 
        default='src/staging/NY-House-Cleaned.csv',  # Default path to the cleaned dataset
        help='Path to the cleaned CSV file (default: src/staging/NY-House-Cleaned.csv)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='src/processed/NY-House-Features.csv',  # Default path for the processed dataset
        help='Path to save the processed dataset (default: src/processed/NY-House-Features.csv)'
    )
    args = parser.parse_args()
    main(args.input, args.output)