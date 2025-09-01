import pandas as pd
import numpy as np
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logger = logging.getLogger(__name__)

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Load and preprocess the house prices dataset.
    
    Parameters
    ----------
    file_path : str
        Path to CSV file

    Returns
    -------
    df : pd.DataFrame
        Preprocessed DataFrame with the TotalLivingArea feature
    """
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded dataset with shape: {df.shape}")

        # Define the columns that contribute to the total living area
        # GrLivArea already includes 1stFlrSF and 2ndFlrSF. This avoids double counting.
        area_columns = ['GrLivArea', 'BsmtFinSF1', 'BsmtFinSF2']

        # Check if all necessary columns exist
        missing_cols = [col for col in area_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")

        # Fill NaN values with 0 for area columns (assuming missing means 0 area)
        df[area_columns] = df[area_columns].fillna(0)

        # Create the TotalLivingArea feature
        df['TotalLivingArea'] = df[area_columns].sum(axis=1)

        # Remove rows with missing SalePrice
        initial_rows = len(df)
        df = df.dropna(subset=['SalePrice'])
        final_rows = len(df)
        
        if initial_rows > final_rows:
            logger.info(f"Removed {initial_rows - final_rows} rows with missing SalePrice")

        logger.info(f"Final shape of the dataset: {df.shape}")
        return df
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate various regression metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    metrics : dict
        Dictionary containing various metrics
    """
    epsilon = 1e-8  # Small constant to avoid division by zero in MAPE
    metrics = {
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    }
    
    return metrics

def display_data_info(df: pd.DataFrame) -> None:
    """Display basic information about the dataset."""
    print("\nDATASET INFORMATION")
    print("="*50)
    print(f"Shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print("\nNUMERICAL STATISTICS:")
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numerical_cols].describe())
    
    print("\nMISSING VALUES:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        # Create a summary DataFrame for missing values
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Count': missing.values,
            '% Missing': (missing / len(df)) * 100
        })
        # Show only columns with missing values
        print(missing_df[missing_df['Missing Count'] > 0].round(2))
    else:
        print("No missing values found!")

def save_model_results(model, filename: str = 'model_results.txt') -> None:
    """Save model results to a text file."""
    params = model.get_params()
    
    with open(filename, 'w') as f:
        f.write("Linear Regression Model Results\n")
        f.write("================================\n\n")
        for key, value in params.items():
            # Convert the value to string to avoid formatting errors with non-numeric types
            f.write(f"{key.capitalize()}: {str(value)}\n")

    logger.info(f"Model results saved to {filename}")