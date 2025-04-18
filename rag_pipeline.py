import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import google.generativeai as genai
import dotenv
import os
import json

# Load environment variables
dotenv.load_dotenv()

# Initialize PostgreSQL connection
conn = psycopg2.connect(
    dbname=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    host=os.getenv('DB_HOST'),
    port=os.getenv('DB_PORT')
)
engine = create_engine(
    f"postgresql+psycopg2://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}")

# Initialize Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))


# Define SQL query function
def sql_query(query: str):
    """Run a SQL SELECT query on the PostgreSQL database and return results."""
    with engine.connect() as connection:
        return pd.read_sql_query(query, connection).to_dict(orient='records')


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
    df = pd.read_sql(query, engine)
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


# Cleanup (optional, call when shutting down)
def cleanup():
    conn.close()
    engine.dispose()
