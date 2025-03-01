import pandas as pd
import logging
import argparse
from sqlalchemy import create_engine

# Setup logging
logging.basicConfig(level=logging.INFO, filename="etl.log", filemode="w",
                    format="%(asctime)s - %(levelname)s - %(message)s")

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

def drop_invalid_sale_price(df):
    """Drop rows with missing or zero sale price."""
    df["SALE PRICE"] = pd.to_numeric(df["SALE PRICE"], errors='coerce')  # Convert to numeric
    df = df[df["SALE PRICE"].notna()]  # Remove NaN values
    df = df[df["SALE PRICE"] > 0]  # Remove 0 values
    logging.info(f"Dropped rows with missing or zero sale price. New dataset has {df.shape[0]} rows.")
    return df


def drop_outliers(df):
    """Drop rows with sale price beyond 3 standard deviations."""
    for col in df.select_dtypes(include=["number"]).columns:  # Apply only to numeric columns
        mean, std = df[col].mean(), df[col].std()
        df = df[(df[col] > mean - 3 * std) & (df[col] < mean + 3 * std)]
    logging.info("Handled outliers.")
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

def load_data(file_path):
    """Load data from CSV."""
    logging.info(f"Loading data from {file_path}")

    try:
        df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded CSV file with {df.shape[0]} rows.")
        return df

    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None

def clean_data(df):
    """Clean and preprocess dataset."""
    logging.info("Starting data cleaning...")

    df = drop_duplicates(df)
    df = drop_invalid_sale_price(df)  # Merged function for NaN and 0
    df = drop_columns_50_missing(df)
    df = drop_outliers(df)

    logging.info(f"Cleaning completed. Final dataset has {df.shape[0]} rows.")
    return df

def transform_data(df):
    """Transform cleaned data."""
    logging.info("Starting data transformation...")

    df = type_conversion(df)
    df = fix_inconsistencies(df)

    logging.info("Transformation completed.")
    return df


def encode_categorical(df):
    """One-hot encode categorical variables and drop original columns."""
    logging.info("Starting categorical encoding...")

    categorical_columns = ["BOROUGH", "BUILDING CLASS CATEGORY", "TAX CLASS AT TIME OF SALE"]

    # Apply one-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    logging.info(f"Categorical encoding completed. New dataset has {df_encoded.shape[1]} columns.")
    return df_encoded




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

    args = parser.parse_args()

    df = load_data(args.input)
    if df is not None:
        df = clean_data(df)  # ✅ Clean first
        df = transform_data(df)  # ✅ Then transform

        save_to_sql(df, "nyc_property_sales", args.db_url)

        df = encode_categorical(df)

        save_data(df, args.output)
