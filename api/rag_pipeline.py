# api/rag_pipeline.py
import pandas as pd
from sqlalchemy import create_engine
import google.generativeai as genai
import os
import json
from contextlib import contextmanager


# Initialize Gemini
def init_genai():
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    genai.configure(api_key=api_key)


# Create a new database connection for each request
@contextmanager
def get_db_connection():
    # Get DB credentials from environment
    db_params = {
        'dbname': os.environ.get('DB_NAME'),
        'user': os.environ.get('DB_USER'),
        'password': os.environ.get('DB_PASSWORD'),
        'host': os.environ.get('DB_HOST'),
        'port': os.environ.get('DB_PORT')
    }

    # Validate DB credentials
    for key, value in db_params.items():
        if not value:
            raise ValueError(f"{key} environment variable not set")

    # Create a connection string
    conn_string = f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"

    # Create a new connection
    engine = create_engine(conn_string)
    conn = engine.connect()

    try:
        yield conn, engine
    finally:
        conn.close()
        engine.dispose()


# Define SQL query function
def sql_query(query: str):
    """Run a SQL SELECT query on the PostgreSQL database and return results."""
    with get_db_connection() as (conn, engine):
        return pd.read_sql_query(query, conn).to_dict(orient='records')


# Get database schema
def get_table_schema():
    table_name = 'change_order_reports'
    query = f"""
    select 
        column_name, 
        data_type
    from information_schema.columns
    where table_schema = 'public' and table_name = '{table_name}'
    order by ordinal_position;
    """
    with get_db_connection() as (conn, engine):
        df = pd.read_sql(query, conn)

    table_schema = {
        "table": table_name,
        "columns": [
            {"name": row['column_name'].lower(), "type": row['data_type']}
            for _, row in df.iterrows()
        ]
    }
    return table_schema


# Initialize Gemini with system prompt and tools
def initialize_gemini():
    init_genai()  # Initialize Gemini API

    table_schema = get_table_schema()
    system_prompt = f"""
    You are an expert SQL analyst. When appropriate, generate SQL queries based on the user question and the database schema.
    When you generate a query, use the 'sql_query' function to execute the query on the database and get the results.
    Then, use the results to answer the user's question.

    database_schema: [{json.dumps(table_schema)}]
    """.strip()

    sql_gemini = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        tools=[sql_query],
        system_instruction=system_prompt
    )
    return sql_gemini


# Main function to process user prompt
def process_prompt(prompt: str) -> str:
    # Initialize Gemini model
    sql_gemini = initialize_gemini()

    # Start chat with automatic function calling
    try:
        chat = sql_gemini.start_chat(enable_automatic_function_calling=True)
        response = chat.send_message(prompt).text
        return response
    except Exception as e:
        return f"Error processing prompt: {str(e)}"