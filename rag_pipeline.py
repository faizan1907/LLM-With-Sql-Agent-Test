# api/index.py
from flask import Flask, request, jsonify
import os
import pandas as pd
from sqlalchemy import create_engine
import json
from contextlib import contextmanager

# Initialize Flask app
app = Flask(__name__)


# Connection management
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
    try:
        # Import here to avoid top-level import issues
        import google.generativeai as genai

        # Initialize Gemini API
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=api_key)

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
    except Exception as e:
        raise Exception(f"Error initializing Gemini: {str(e)}")


# Main function to process user prompt
def process_prompt(prompt: str) -> str:
    try:
        sql_gemini = initialize_gemini()
        chat = sql_gemini.start_chat(enable_automatic_function_calling=True)
        response = chat.send_message(prompt).text
        return response
    except Exception as e:
        return f"Error processing prompt: {str(e)}"


# API routes
@app.route('/api/query', methods=['POST'])
def query_excel():
    # Check if request contains JSON data
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    # Extract prompt from JSON
    data = request.get_json()
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({"error": "Missing 'prompt' in request body"}), 400

    try:
        # Process the prompt using the RAG pipeline
        result = process_prompt(prompt)
        return jsonify({"response": result}), 200
    except Exception as e:
        return jsonify({"error": f"Error processing prompt: {str(e)}"}), 500


# Default route
@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "API is running", "endpoints": ["/api/query"]}), 200


# For local development only
if __name__ == '__main__':
    app.run(debug=True)
