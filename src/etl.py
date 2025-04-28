import argparse
import pandas as pd
import os

def process_csv(input_path: str, output_path: str):
    """
    Process a raw CSV file: clean it and save the cleaned data.
    """
    # Check if input file exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"The input file does not exist: {input_path}")

    # Read raw dataset
    df = pd.read_csv(input_path, encoding='utf-8', low_memory=False)

    # Select relevant columns
    cols = ['BROKERTITLE', 'TYPE', 'PRICE', 'BEDS', 'BATH', 'PROPERTYSQFT',
            'ADDRESS', 'STATE', 'MAIN_ADDRESS', 'ADMINISTRATIVE_AREA_LEVEL_2',
            'LOCALITY', 'SUBLOCALITY', 'STREET_NAME', 'LONG_NAME',
            'FORMATTED_ADDRESS', 'LATITUDE', 'LONGITUDE']

    # Ensure the selected columns exist in the dataset
    missing_cols = [col for col in cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"The following required columns are missing in the dataset: {missing_cols}")

    df = df[cols]

    # Drop duplicates and handle missing values
    df = df.drop_duplicates()
    df = df.dropna(subset=['PRICE', 'BEDS', 'BATH', 'PROPERTYSQFT', 'ADDRESS'])

    # Create staging folder if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write cleaned data to output
    df.to_csv(output_path, index=False)
    print(f"Cleaned data saved to {output_path}")

def main():
    """
    Main function to process the CSV file.
    """
    parser = argparse.ArgumentParser(description="ETL script for real estate data.")
    parser.add_argument(
        "--input", 
        type=str, 
        default="src/raw/NY-House-Dataset.csv",  # Default input path
        help="Path to raw CSV file (default: src/raw/NY-House-Dataset.csv)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="src/staging/NY-House-Cleaned.csv",  # Default output path
        help="Path to save cleaned CSV (default: src/staging/NY-House-Cleaned.csv)"
    )
    args = parser.parse_args()

    # Process the CSV file
    process_csv(args.input, args.output)

if __name__ == "__main__":
    main()