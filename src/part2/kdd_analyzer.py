import os
import argparse
import sys
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def analyze_dataset(file_path: str) -> None:
    """
    Reads a dataset from the provided path and performs basic analysis.
    
    Args:
        file_path (str): The path to the CSV dataset.
    """
    try:
        print(f"Loading dataset from: {file_path}")
        # Make sure the file exists before reading
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
            
        dataset = pd.read_csv(file_path)
        
        print("\nDataset Overview:")
        print("-" * 30)
        print(f"Shape (Rows, Columns): {dataset.shape}")
        print("\nFirst 5 Rows:")
        print(dataset.head())
        
        print("\nSummary Statistics:")
        print(dataset.describe(include='all').to_string())
        
    except FileNotFoundError as fnf:
        print(f"File Error: {fnf}")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("Data Error: The provided file is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze the KDD dataset.")
    parser.add_argument(
        "--data_path", 
        type=str, 
        help="Path to the dataset CSV file. Defaults to DATASET_PATH from .env or local data folder."
    )
    
    args = parser.parse_args()
    
    # Precedence: 1. CLI Arg, 2. Env Var, 3. Default Relative Path
    default_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'Sample_KDD.csv')
    dataset_path = args.data_path or os.getenv("DATASET_PATH") or os.path.abspath(default_path)
    
    analyze_dataset(dataset_path)
