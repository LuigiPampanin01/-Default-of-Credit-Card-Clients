import pandas as pd
import logging
import argparse
from sqlalchemy import create_engine
from sklearn.preprocessing import OrdinalEncoder
import numpy as np


# Setup logging
logging.basicConfig(level=logging.INFO, filename="etl.log", filemode="w",
                    format="%(asctime)s - %(levelname)s - %(message)s")

def unnamed_columns(df):
    # Drop the Unnamed: 0 column if it exists
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
        logging.info(f"Dropped 'Unnamed: 0' column. Shape of dataset: {df.shape[0]} rows and {df.shape[1]} columns.")
    return df



def drop_duplicates(df):
    """Drop duplicate rows."""

    df.drop_duplicates(inplace=True)
    logging.info(f"Dropped duplicates. New dataset has {df.shape[0]} rows.")
    return df

def drop_columns_50_missing(df):
    """Drop columns with more than 50% missing values."""
    df = df.dropna(thresh=df.shape[0] * 0.5, axis=1)
    logging.info(f"Dropped columns with more than 50% missing values. New dataset has {df.shape[1]} columns.")
    return df


def type_conversion(df):
    """Convert columns to appropriate data types."""
    df["SALE DATE"] = pd.to_datetime(df["SALE DATE"], errors='coerce')  # Convert dates

    # Convert numeric columns
    df["SALE PRICE"] = pd.to_numeric(df["SALE PRICE"], errors='coerce')
    df["LAND SQUARE FEET"] = pd.to_numeric(df["LAND SQUARE FEET"], errors='coerce')
    df["GROSS SQUARE FEET"] = pd.to_numeric(df["GROSS SQUARE FEET"], errors='coerce')

    return df

def fix_inconsistencies(df):
    """Standardize categorical values (e.g., Borough names)."""
    df["BOROUGH"] = df["BOROUGH"].astype(str)  # Convert to string

    # Standardize borough names
    borough_mapping = {
        "1": "manhattan",
        "2": "bronx",
        "3": "brooklyn",
        "4": "queens",
        "5": "staten island"
    }
    df["BOROUGH"] = df["BOROUGH"].replace(borough_mapping)

    return df

def drop_reduntant_columns(df):
    red_cols = ["EASE-MENT", "APARTMENT NUMBER", "ADDRESS", "LOT",  "BUILDING CLASS AT TIME OF SALE", "NEIGHBORHOOD", "BUILDING CLASS AT PRESENT"]
    df = df.drop(red_cols, axis=1)  # Drop columns with no useful information

    logging.info(f'Dropped columns with no useful information {red_cols}. New dataset has {df.shape[1]} columns.')

    return df


def drop_invalid_sale_price(df):
    """Drop rows with missing, zero, or unrealistically low sale prices."""
    df["SALE PRICE"] = pd.to_numeric(df["SALE PRICE"], errors='coerce')  # Ensure numeric type
    df = df[df["SALE PRICE"].notna()]  # Remove NaN values
    df = df[df["SALE PRICE"] > 0]  # Remove zero sales

    # Define threshold for unrealistic sales
    low_price_threshold = 50000  # Adjust as needed
    high_price_threshold = 1e8  # Adjust as needed
    df = df[(df["SALE PRICE"] >= low_price_threshold) & (df["SALE PRICE"] <= high_price_threshold)]  # Keep only valid transactions

    logging.info(f"Dropped rows with missing, zero, or unrealistically low sale price. New dataset has {df.shape[0]} rows.")
    return df

import pandas as pd
import numpy as np
import logging

def remove_outliers(df):
    """
    Removes rows where any numeric column has values beyond 3 standard deviations from the mean.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The cleaned DataFrame with outliers removed.
    """
    # Select only numeric columns dynamically
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Initialize a mask with all True values (keep all rows initially)
    mask = np.ones(df.shape[0], dtype=bool)

    # Compute mean and standard deviation for each numeric column
    for col in numeric_cols:
        mean = df[col].mean()
        std = df[col].std()

        # Define lower and upper limits (-3σ and +3σ)
        lower_limit = mean - 3 * std
        upper_limit = mean + 3 * std

        # Update the mask to filter outliers
        mask &= (df[col] >= lower_limit) & (df[col] <= upper_limit)

    # Apply mask to remove outliers
    df_cleaned = df[mask]

    logging.info(f"Removed outliers. New dataset has {df_cleaned.shape[0]} rows.")

    return df_cleaned


import logging
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def encode_categorical(df):
    """Applies categorical encoding: One-Hot, Ordinal, and Target Encoding."""
    logging.info("Starting categorical encoding...")

    # 🔹 One-Hot Encoding for BOROUGH 
    categorical_columns = ["BOROUGH"]
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    logging.info(f"One-Hot encoding completed. New dataset has {df.shape[1]} columns.")

    # 🔹 Ordinal Encoding for TAX CLASS AT PRESENT
    df["TAX CLASS AT PRESENT"] = df["TAX CLASS AT PRESENT"].astype(str).str.strip()
    df["TAX CLASS AT PRESENT"].replace("", "Unknown", inplace=True)
    df["TAX CLASS AT PRESENT"].fillna("Unknown", inplace=True)

    tax_class_order = [["Unknown", "1", "1A", "1B", "1C", "2", "2A", "2B", "2C", "4"]]
    ordinal_encoder = OrdinalEncoder(categories=tax_class_order, handle_unknown="use_encoded_value", unknown_value=-1)
    df["TAX_CLASS_ENCODED"] = ordinal_encoder.fit_transform(df[["TAX CLASS AT PRESENT"]])

    # Drop original column
    df.drop(columns=["TAX CLASS AT PRESENT"], inplace=True)

    logging.info(f"Ordinal encoding completed. New dataset has {df.shape[1]} columns.")

    # 🔹 Convert `SALE DATE` to Numeric Features (Year, Month)
    df["SALE DATE"] = pd.to_datetime(df["SALE DATE"], errors="coerce")
    df["SALE_YEAR"] = df["SALE DATE"].dt.year
    df["SALE_MONTH"] = df["SALE DATE"].dt.month
    df.drop(columns=["SALE DATE"], inplace=True)

    logging.info("Extracted numerical features from SALE DATE.")

    # 🔹 Standardize One-Hot Encoded Column Names (Remove Extra Spaces)
    df.columns = df.columns.str.strip()

    logging.info(f"Final dataset has {df.shape[1]} columns after encoding.")
    return df




def load_data(file_path):
    """Load data from CSV."""
    logging.info(f"Loading data from {file_path}")

    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded CSV file with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df

    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def clean_data(df):
    """Clean and preprocess dataset."""
    logging.info("Starting data cleaning...")

    df = unnamed_columns(df)
    df = drop_duplicates(df)
    df = drop_invalid_sale_price(df)  # Merged function for NaN and 0 or low sale prices
    df = remove_outliers(df)
    df = drop_columns_50_missing(df)
    df = drop_reduntant_columns(df)
    
    logging.info(f"Cleaning completed. Final dataset has {df.shape[0]} rows and {df.shape[1]} columns.")
    return df

def transform_data(df):
    """Transform cleaned data."""
    logging.info("Starting data transformation...")

    df = type_conversion(df)
    df = fix_inconsistencies(df)

    logging.info("Transformation completed.")
    return df

def save_data(df, output_path):
    """Save cleaned data to CSV."""
    df.to_csv(output_path, index=False)
    logging.info(f"Cleaned data saved to {output_path}")


def save_to_sql(df, table_name, db_url):
    """
    Save the cleaned DataFrame to an SQL database.
    
    :param df: Cleaned pandas DataFrame
    :param table_name: Name of the SQL table
    :param db_url: Database connection URL
    """
    try:
        logging.info(f"Connecting to database")
        
        # Create SQLAlchemy engine
        engine = create_engine(db_url)

        # Save DataFrame to SQL table
        df.to_sql(table_name, con=engine, if_exists="replace", index=False)

        logging.info(f"Data successfully saved to table '{table_name}' in database.")


    except Exception as e:
        logging.error(f"Error saving to SQL: {e}")
 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse and clean NYC Property Sales data.")
    parser.add_argument("--input", required=True, help="Path to input file")
    parser.add_argument("--output", required=True, help="Path to output CSV file")
    parser.add_argument("--db_url", required=True, help="SQL database connection URL")
    parser.add_argument("--output_sql_csv", required=True, help="Path to output CSV file to save the data like the data in SQL")

    args = parser.parse_args()

    df = load_data(args.input)
    if df is not None:
        df = clean_data(df)  
        df = transform_data(df)  

        save_to_sql(df, "nyc_property_sales", args.db_url)
        save_data(df, args.output_sql_csv)

        df.drop(columns=["TOTAL UNITS"], inplace=True) # dropped for redudancy
        df.drop(columns=["ZIP CODE"], inplace=True)
        logging.info("Dropped 'TOTAL UNITS' column for redundancy.")

        df = encode_categorical(df)

        logging.info(f"Final dataset shape: {df.shape[0]} rows and {df.shape[1]} columns.")

        save_data(df, args.output)
        logging.info("ETL process completed.")
    else:
        logging.error("ETL process failed. Check the logs for more information.")