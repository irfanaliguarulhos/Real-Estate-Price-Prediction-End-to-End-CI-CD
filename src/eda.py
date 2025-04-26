import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os

def run_eda(input_path: str, output_dir: str):
    """
    Perform Exploratory Data Analysis (EDA) on the dataset and save visualizations.
    Args:
        input_path (str): Path to the cleaned CSV file.
        output_dir (str): Directory to save the EDA plots.
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Check if the input file exists
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"The input file does not exist: {input_path}")

    # Read cleaned dataset
    df = pd.read_csv(input_path)

    # Check required columns exist in the dataset
    required_columns = ['PRICE', 'BEDS', 'STATE']  # Updated column names to match your dataset
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"The following required columns are missing in the dataset: {missing_columns}")

    # Price distribution
    plt.figure()
    df['PRICE'].hist(bins=50)
    plt.title('Price Distribution')
    plt.xlabel('Price')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(output_dir, 'price_dist.png'))
    plt.close()

    # Bedrooms vs Price
    plt.figure()
    df.boxplot(column='PRICE', by='BEDS')
    plt.title('Price by Bedrooms')
    plt.xlabel('Bedrooms')
    plt.ylabel('Price')
    plt.savefig(os.path.join(output_dir, 'price_bedrooms.png'))
    plt.close()

    # State counts
    plt.figure()
    df['STATE'].value_counts().plot(kind='bar')
    plt.title('Properties per State')
    plt.xlabel('State')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'state_counts.png'))
    plt.close()

    print(f"EDA plots saved in {output_dir}")

if __name__ == '__main__':
    # Argument parser for CLI inputs
    parser = argparse.ArgumentParser(description="EDA script for real estate data.")
    parser.add_argument(
        '--input', 
        type=str, 
        default='src/staging/NY-House-Cleaned.csv',  # Default path to the cleaned dataset
        help='Path to the cleaned CSV file (default: src/staging/NY-House-Cleaned.csv)'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='src/eda_output',  # Default directory for saving EDA plots
        help='Directory to save the EDA visualizations (default: src/eda_output)'
    )
    args = parser.parse_args()

    # Run EDA with the provided or default paths
    run_eda(args.input, args.output)
