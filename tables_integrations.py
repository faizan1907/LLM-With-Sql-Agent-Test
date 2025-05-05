from contextlib import contextmanager

import pandas as pd
import os
import glob
import logging
import re  # For sanitizing names
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import create_engine  # Need this explicitly for get_db_connection if not already imported elsewhere
from dotenv import load_dotenv  # Optional: for loading env vars from a .env file

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load Environment Variables (Optional but Recommended) ---
# Create a .env file in the same directory as the script with your database credentials
# e.g.,
# DB_NAME=your_db_name
# DB_USER=your_db_user
# DB_PASSWORD=your_db_password
# DB_HOST=your_db_host
# DB_PORT=your_db_port
load_dotenv()


# --- Your get_db_connection Context Manager (Include this function) ---
# Make sure this function is defined and accessible
# Assuming os and create_engine are imported where this function is defined
@contextmanager
def get_db_connection():
    """Provides a managed database connection."""
    db_params = {
        'dbname': os.environ.get('DB_NAME'),
        'user': os.environ.get('DB_USER'),
        'password': os.environ.get('DB_PASSWORD'),
        'host': os.environ.get('DB_HOST'),
        'port': os.environ.get('DB_PORT')
    }
    if not all(db_params.values()):
        raise ValueError(
            "Database connection parameters (DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT) must be set in environment variables.")

    # Construct the connection string. Using f-string for clarity.
    # Ensure proper quoting if passwords contain special characters, though psycopg2+SQLAlchemy usually handle this.
    # For maximum safety with special characters in passwords, you might need urllib.parse.quote_plus
    # Example: user:pass@host:port/db -> user:quoted_pass@host:port/db
    from urllib.parse import quote_plus
    conn_string = f"postgresql+psycopg2://{db_params['user']}:{quote_plus(db_params['password'])}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"

    engine = None
    conn = None
    try:
        logger.debug("Attempting to connect to the database.")
        engine = create_engine(conn_string)
        # Use engine.connect() to get a connection object
        conn = engine.connect()
        logger.debug("Database connection successful.")
        # Yield both connection and engine if needed elsewhere, though only conn is strictly necessary for to_sql
        yield conn, engine
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}", exc_info=True)
        raise  # Re-raise the exception after logging
    finally:
        # Ensure resources are cleaned up
        if conn:
            conn.close()
            logger.debug("Database connection closed.")
        if engine:
            # Dispose the engine to release all connections in the pool
            engine.dispose()
            logger.debug("Database engine disposed.")


# --- Helper function to sanitize names for database identifiers ---
def sanitize_name(name):
    """Sanitizes a string to be a valid SQL table or column name."""
    # Convert to lowercase
    name = name.lower()
    # Replace spaces and hyphens with underscores
    name = name.replace(" ", "_").replace("-", "_")
    # Remove any characters that are not alphanumeric or underscore
    name = re.sub(r'[^\w]+', '', name)
    # Ensure it doesn't start with an underscore (some SQL dialects dislike this)
    name = name.lstrip('_')
    # If the name became empty or is just underscores, use a default or raise error
    if not name:
        # Or raise ValueError("Cannot sanitize name into a valid identifier")
        return "invalid_name"  # Or handle appropriately
    # Ensure it doesn't start with a digit (some SQL dialects dislike this)
    if name[0].isdigit():
        name = "_" + name
    # SQL identifiers have length limits, but sanitization usually keeps it shorter.
    # Consider truncating if names are excessively long, depending on your DB.
    # Example for PostgreSQL: max 63 bytes
    # return name[:63] # Optional: Add truncation if needed

    return name


# --- Function to process a single Excel file ---
def process_excel_file(filepath):
    """Reads an Excel file, creates tables, and inserts data."""
    logger.info(f"Processing file: {filepath}")
    inserted_counts = {}
    filename_base = os.path.splitext(os.path.basename(filepath))[0]

    try:
        # Read all sheets from the Excel file
        excel_data = pd.read_excel(filepath, sheet_name=None)

        for sheet_name, df in excel_data.items():
            logger.info(f"  Processing sheet: '{sheet_name}'")

            if df.empty:
                logger.warning(f"    Sheet '{sheet_name}' is empty. Skipping.")
                inserted_counts[sheet_name] = 0
                continue

            # Sanitize sheet name and combine with filename for table name
            sanitized_sheet_name = sanitize_name(sheet_name)
            sanitized_filename_base = sanitize_name(filename_base)
            # Create a unique table name. Using filename_sheetname is common.
            table_name = f"{sanitized_filename_base}_{sanitized_sheet_name}"
            # Truncate table name if necessary (e.g., PostgreSQL max 63 bytes)
            table_name = table_name[:63].rstrip('_')  # Simple truncation

            # Sanitize column names in the DataFrame
            df.columns = [sanitize_name(col) for col in df.columns]

            # Use the context manager to get a database connection
            try:
                with get_db_connection() as (conn, engine):
                    logger.info(f"    Attempting to create/replace table '{table_name}' and insert data.")
                    # Use pandas to_sql to create the table and insert data
                    # if_exists='replace' will drop the table if it exists and create a new one
                    # if_exists='fail' will raise a ValueError if the table exists
                    # if_exists='append' will insert data into the existing table
                    # We use 'replace' as the request implies creating *new* tables from the sheets.
                    # BE CAREFUL with 'replace' as it causes data loss if the table exists!
                    # If you want to avoid accidental overwrite, use 'fail' and handle the table dropping manually beforehand.
                    rows_inserted = df.to_sql(
                        name=table_name,
                        con=conn,
                        if_exists='replace',  # Options: 'fail', 'replace', 'append'
                        index=False  # Don't write DataFrame index as a column
                    )
                    inserted_counts[sheet_name] = rows_inserted
                    logger.info(f"    Successfully inserted {rows_inserted} rows into table '{table_name}'.")

            except SQLAlchemyError as db_err:
                logger.error(
                    f"    ❌ Database error processing sheet '{sheet_name}' into table '{table_name}': {db_err}",
                    exc_info=True)
                inserted_counts[sheet_name] = 0  # Indicate failure for this sheet
            except Exception as e:
                logger.error(
                    f"    ❌ An unexpected error occurred during database operation for sheet '{sheet_name}': {e}",
                    exc_info=True)
                inserted_counts[sheet_name] = 0  # Indicate failure for this sheet


    except FileNotFoundError:
        logger.error(f"❌ Error: File not found at {filepath}")
        inserted_counts = {"Error": 0}  # Indicate file error
    except pd.errors.EmptyDataError:
        logger.warning(f"⚠️ Warning: Excel file is empty or corrupt: {filepath}")
        inserted_counts = {"Empty File": 0}  # Indicate empty file
    except Exception as e:
        logger.error(f"❌ An unexpected error occurred processing file {filepath}: {e}", exc_info=True)
        inserted_counts = {"Error": 0}  # Indicate general error

    return inserted_counts


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- Configuration ---
    excel_files_directory = './excel_files'  # <--- !!! CHANGE THIS TO YOUR DIRECTORY !!!

    # Ensure the directory exists
    if not os.path.isdir(excel_files_directory):
        logger.error(f"Error: Directory not found: {excel_files_directory}")
        exit(1)

    # Find all .xlsx files in the directory
    xlsx_files = glob.glob(os.path.join(excel_files_directory, '*.xlsx'))

    if not xlsx_files:
        logger.warning(f"No .xlsx files found in directory: {excel_files_directory}")
        exit(0)

    logger.info(f"Found {len(xlsx_files)} Excel files to process.")

    total_rows_processed = 0
    all_results = {}  # Dictionary to store results per file/sheet

    for file_path in xlsx_files:
        file_results = process_excel_file(file_path)
        all_results[os.path.basename(file_path)] = file_results
        # Sum up rows inserted for reporting (only count successful inserts)
        total_rows_processed += sum(count for count in file_results.values() if isinstance(count, int))

    # --- Final Summary ---
    logger.info("\n--- Processing Summary ---")
    for filename, sheet_results in all_results.items():
        logger.info(f"File: {filename}")
        for sheet_name, row_count in sheet_results.items():
            if isinstance(row_count, int):
                logger.info(f"  Sheet '{sheet_name}': {row_count} rows inserted.")
            else:
                logger.info(f"  Sheet '{sheet_name}': Processing failed or skipped.")

    logger.info(f"\nTotal rows attempted for insertion across all files and sheets: {total_rows_processed}")
    logger.info("--- End of Summary ---")
