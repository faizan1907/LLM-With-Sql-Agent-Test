# rag_pipeline.py
import os
import pandas as pd
from sqlalchemy import create_engine, text, inspect
import json
from contextlib import contextmanager
import re
import logging
import google.generativeai as genai
from dotenv import load_dotenv
import ast
from typing import List

# Load environment variables from .env file
load_dotenv()

# Configure logging with a cleaner format for presentation
log_format = '%(message)s'  # Remove timestamp for cleaner output
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger(__name__)


# --- Database Connection ---
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
            "Database connection parameters (DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT) must be set in "
            "environment variables.")

    conn_string = f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}"
    engine = None
    conn = None
    try:
        logger.debug("Attempting to connect to the database.")
        # Increase statement_timeout for potentially longer schema introspection
        engine = create_engine(conn_string, connect_args={'options': '-c statement_timeout=30000'})  # 30 seconds
        conn = engine.connect()
        logger.debug("Database connection successful.")
        yield conn, engine
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}", exc_info=True)
        raise
    finally:
        if conn:
            conn.close()
            logger.debug("Database connection closed.")
        if engine:
            engine.dispose()
            logger.debug("Database engine disposed.")


def sql_query(query: str):
    """
    Runs a SQL SELECT query on the PostgreSQL database and returns results as a list of dictionaries.
    """
    logger.debug(f"Executing SQL query: {query}")
    try:
        with get_db_connection() as (conn, engine):
            print(f"query: {query}")  # Keep this print for debugging generated SQL
            result = pd.read_sql_query(text(query), conn)
            # Convert NaN/NaT to None for JSON compatibility
            result = result.where(pd.notnull(result), None)
            data = result.to_dict(orient='records')
            logger.debug(f"Query executed successfully. Rows returned: {len(data)}")
            if data:
                # Log only a small sample if data is large
                sample_size = min(len(data), 2)
                logger.debug(f"Query result sample: {json.dumps(data[:sample_size], indent=2, default=str)}")
            else:
                logger.debug("Query returned no data.")
            return data
    except Exception as e:
        logger.error(f"❌ Error executing SQL query: {e}", exc_info=True)
        # Optionally return an error structure instead of raising
        # return {"error": f"SQL execution failed: {e}"}
        raise


def get_database_schema(table_names_to_fetch: List[str]):
    """
    Retrieves the schema (column names and types) for a specified list of tables.

    Args:
        table_names_to_fetch: A list of table names (strings) to get schemas for.

    Returns:
        A JSON string representing the schemas of the requested tables,
        or an error message string starting with "Error:" if a major failure occurs.
        Returns "{}" if the input list is empty or none of the tables are found.
    """
    if not table_names_to_fetch:
        logger.warning("No table names provided to get_database_schema. Returning empty schema.")
        return "{}"

    logger.debug(f"Retrieving schema for specified tables: {table_names_to_fetch}")
    specified_schemas = {}
    try:
        with get_db_connection() as (conn, engine):
            inspector = inspect(engine)
            default_schema = inspector.default_schema_name  # Assuming tables are in the default schema
            logger.debug(f"Inspecting schema: {default_schema}")

            found_tables_count = 0
            for table_name in table_names_to_fetch:
                try:
                    # Check if table exists before trying to get columns
                    if not inspector.has_table(table_name, schema=default_schema):
                        logger.warning(f"[!] Requested table '{table_name}' not found in schema '{default_schema}'. "
                                       f"Skipping.")
                        continue  # Skip to the next table name in the list

                    columns = inspector.get_columns(table_name, schema=default_schema)
                    # Store schema as {column_name: type_string}
                    specified_schemas[table_name] = {col['name']: str(col['type']) for col in columns}
                    logger.debug(f"Schema retrieved for table '{table_name}'.")
                    found_tables_count += 1
                except Exception as e:
                    logger.error(f"❌ Error retrieving columns for table '{table_name}': {e}", exc_info=False)
                    specified_schemas[table_name] = {"error": f"Could not retrieve schema: {e}"}

            if found_tables_count == 0:
                logger.warning(
                    f"None of the requested tables ({table_names_to_fetch}) were found. Returning empty schema.")
                return "{}"

            logger.debug(
                f"Schema retrieval complete for specified tables. "
                f"Found {found_tables_count}/{len(table_names_to_fetch)}.")
            # Convert the dictionary of schemas to a JSON string
            return json.dumps(specified_schemas, indent=2)

    except Exception as e:
        logger.error(f"❌ Error retrieving database schema for tables {table_names_to_fetch}: {e}", exc_info=True)
        # Return a clear error message
        return f"Error: Failed to retrieve database schema: {e}"


def initialize_gemini_model(model_name="gemini-1.5-flash", system_instruction=None):
    """Initializes and configures the Gemini model."""
    api_key = os.environ.get('GEMINI_API_KEY')
    if not api_key:
        raise ValueError("GEMINI_API_KEY must be set in environment variables.")

    genai.configure(api_key=api_key)

    generation_config = {
        "temperature": 0.1,  # Lower temperature might help with consistency
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]

    model_kwargs = {
        "model_name": model_name,
        "safety_settings": safety_settings,
        "generation_config": generation_config,
    }
    if system_instruction:
        model_kwargs["system_instruction"] = system_instruction

    try:
        model = genai.GenerativeModel(**model_kwargs)
        logger.debug(f"Gemini model '{model_name}' initialized successfully.")
        return model
    except Exception as e:
        logger.error(f"❌ Failed to initialize Gemini model '{model_name}': {e}", exc_info=True)
        raise


def clean_response_text(text):
    """Removes markdown code blocks and trims whitespace."""
    # Remove ```sql, ```json, ```python etc.
    text = re.sub(r'^```[a-zA-Z]*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?```$', '', text, flags=re.MULTILINE)
    return text.strip()


def parse_tasks_response(response_text):
    """
    Safely parse the task decomposition response into a Python list of dictionaries.
    Handles JSON null values by converting them to Python None.
    """
    try:
        # First try ast.literal_eval after replacing null
        python_compatible = response_text.replace('null', 'None')
        tasks = ast.literal_eval(python_compatible)
        if not isinstance(tasks, list):
            raise ValueError("Decomposition did not return a list (evaluated by ast).")
        # Basic validation of task structure
        for task in tasks:
            if not isinstance(task, dict):
                raise ValueError("Task item is not a dictionary.")
            if 'task_type' not in task or 'description' not in task:
                logger.warning(f"Task missing 'task_type' or 'description': {task}")
        return tasks
    except (SyntaxError, ValueError, TypeError) as ast_error:
        logger.warning(f"AST parsing failed: {ast_error}. Trying JSON parsing.")
        try:
            # If ast fails, try json.loads
            tasks = json.loads(response_text)
            if not isinstance(tasks, list):
                raise ValueError("Decomposition did not return a list (evaluated by json).")
            # Basic validation of task structure
            for task in tasks:
                if not isinstance(task, dict):
                    raise ValueError("Task item is not a dictionary.")
                if 'task_type' not in task or 'description' not in task:
                    logger.warning(f"Task missing 'task_type' or 'description': {task}")
            return tasks
        except (json.JSONDecodeError, ValueError, TypeError) as json_error:
            logger.error(
                f"Failed to parse task list. Raw response: '{response_text}'. AST error: {ast_error}. "
                f"JSON error: {json_error}",
                exc_info=True)
            raise ValueError(
                f"Failed to parse task list. Check AI response format. AST: {ast_error}, JSON: {json_error}")


# --- Core Processing Logic ---
def process_prompt(prompt: str, target_tables: List[str]):
    """
    Processes a user prompt to generate insights, visualizations, or reports, using the schema
    of a specified list of database tables.

    Args:
        prompt: The user's natural language request.
        target_tables: A list of table names (strings) to consider for this request.
    """
    # --- Step 1: Receive and Log Prompt & Target Tables ---
    logger.info("\n✨ STEP 1: PROCESSING USER PROMPT")
    logger.info(f"Received Prompt: \"{prompt}\"")
    if not target_tables:
        logger.error("❌ No target tables specified for processing.")
        return [{"type": "text", "data": "Error: No target tables were specified for the analysis."}]
    logger.info(f"Target Tables: {target_tables}")

    results = []

    try:
        # --- Step 2a: Get Schema for Specified Tables ---
        logger.info("\n📜 STEP 2a: FETCHING SCHEMA FOR SPECIFIED TABLES")
        database_schema_json_or_error = get_database_schema(target_tables)

        if database_schema_json_or_error.startswith("Error:"):
            logger.error(f"❌ Schema retrieval failed: {database_schema_json_or_error}")
            return [{"type": "text", "data": f"Failed to proceed: {database_schema_json_or_error}"}]

        try:
            schema_dict = json.loads(database_schema_json_or_error)
            if not schema_dict:
                logger.warning(
                    f"Database schema is empty. None of the requested tables ({target_tables}) were found or "
                    f"accessible.")
                return [{"type": "text",
                         "data": f"Failed to proceed: None of the specified tables ({target_tables}) "
                                 f"could be found or accessed."}]
        except json.JSONDecodeError:
            logger.error(f"❌ Failed to parse the retrieved schema JSON: {database_schema_json_or_error}")
            return [{"type": "text", "data": "Failed to proceed: Error parsing database schema information."}]

        database_schema_json = database_schema_json_or_error
        logger.info("[✓] Database schema retrieved for specified tables.")

        logger.info("\n🧠 STEP 2b: DECOMPOSING PROMPT INTO TASKS (using provided schema)")

        decomposition_instruction = f""" Analyze the user's request and identify the specific data analysis tasks 
        required. These tasks can be generating a textual insight, creating a visualization (bar or line graph), 
        or generating a structured data report (table). Based ONLY on the database schema provided below (for the 
        specified tables) and the user prompt, list the tasks. Consider if multiple tables FROM THIS LIST need to be 
        combined (joined) to fulfill the request. Do not assume other tables exist.

        Provided Database Schema (Only these tables are available):
        {database_schema_json}

        User Prompt:
        "{prompt}"

        For each task, specify: 1.  'task_type': Either 'insight', 'visualization', or 'report'. Choose 'report' if 
        the user asks for data listed in rows/columns, a table, or a structured list of items with specific 
        attributes (like the example image showing PMs with metrics). 2.  'description': A brief description of what 
        the task aims to achieve (e.g., "Generate report of jobs written and change order size per PM", "Visualize 
        total cost per manager"). 3.  'required_data_summary': Briefly describe the data needed, mentioning the key 
        columns and tables FROM THE PROVIDED SCHEMA needed to construct the insight, visualization, or report. 4.  
        'visualization_type': If task_type is 'visualization', specify 'bar' or 'line'. Otherwise, null.

        Output the result as a valid Python list of dictionaries ONLY. Do not include explanations or markdown. 
        Example: [ {{"task_type": "report", "description": "Report showing number of jobs written and change order 
        size for each project manager", "required_data_summary": "PM name, count of jobs written, sum/avg of change 
        order size, potentially joined from jobs and managers tables (if both provided in schema)", 
        "visualization_type": null}}, {{"task_type": "visualization", "description": "Total estimated vs. actual cost 
        per project manager", "required_data_summary": "manager name, sum of estimated cost, sum of actual cost, 
        joined from projects and managers tables (if both provided in schema)", "visualization_type": "bar"}}, 
        {{"task_type": "insight", "description": "Which project had the biggest cost overrun percentage?", 
        "required_data_summary": "project name, estimated cost, actual cost from projects table (if provided in 
        schema)", "visualization_type": null}} ]"""

        decomposer_model = initialize_gemini_model()
        decomposer_chat = decomposer_model.start_chat()
        response = decomposer_chat.send_message(decomposition_instruction)
        cleaned_response = clean_response_text(response.text)
        logger.debug(f"Raw task decomposition response: {response.text}")
        logger.debug(f"Cleaned task decomposition response: {cleaned_response}")

        try:
            tasks = parse_tasks_response(cleaned_response)
            if not tasks:
                logger.warning("AI task decomposition returned an empty list. No tasks to perform.")
                return [{"type": "text",
                         "data": f"I couldn't identify specific tasks from your request based on the provided "
                                 f"tables ({target_tables}). Could you please rephrase or check the tables?"}]

            logger.info(f"🤖 AI identified {len(tasks)} tasks:")
            for idx, task_item in enumerate(tasks):
                logger.info(
                    f"  • Task {idx + 1}: {task_item.get('description', 'N/A')} ({task_item.get('task_type', 'N/A')})")
        except (ValueError, TypeError) as e:
            logger.error(f"❌ Failed to parse AI task decomposition: {e}", exc_info=True)
            return [{"type": "text",
                     "data": f"Error: Could not understand the tasks required by the prompt. Please rephrase. ("
                             f"Parsing error: {e})"}]

        # --- Step 3: Process Each Task ---
        logger.info("\n⚙️ STEP 3: PROCESSING TASKS")
        sql_gemini = None
        plotly_gemini = None
        insight_gemini = None

        # --- MODIFIED SQL INSTRUCTION ---
        sql_instruction = f""" You are an expert PostgreSQL query writer. Generate a SINGLE, syntactically correct 
        PostgreSQL SELECT query based ONLY on the schema provided and the task description.

        **=== DATABASE SCHEMA (Use ONLY this) ===**
        {database_schema_json}
        
        **=== CRITICAL JOINING RULES ===**
        
        1.  **Primary Entity First:** Identify the main subject (e.g., employees, departments) and start the `FROM` 
        clause with that table.
        
        2.  **`LEFT JOIN` for Completeness:** If the goal is to show *all* items from the primary entity (e.g., 
        "list all departments and their employees"), ALWAYS use `LEFT JOIN` from the primary entity table to others. 
        `INNER JOIN` is WRONG here as it loses primary entities without matches.
        
        3.  **`ON` CLAUSE MUST BE SIMPLE - THE MOST IMPORTANT RULE:** * **JOIN `ON` ONLY THE SINGLE PRIMARY/FOREIGN 
        KEY PAIR.** * Example: `... FROM table_a JOIN table_b ON table_a.id = table_b.a_id ...` (This is CORRECT) * * 
        **--- WARNING: DO NOT ADD MULTIPLE CONDITIONS IN `ON` ---** * **--- WARNING: DO NOT ADD CONDITIONS ON OTHER 
        COLUMNS IN `ON` ---** * * **WRONG:** `ON table_a.id = table_b.a_id AND table_a.status = table_b.status` <== 
        **ABSOLUTELY FORBIDDEN!** * **WRONG:** `ON table_a.id = table_b.a_id AND table_a.type = 'X'` <== **ABSOLUTELY 
        FORBIDDEN!** * **WRONG:** `ON table_a.key1 = table_b.fkey1 AND table_a.key2 = table_b.fkey2` <== **FORBIDDEN! 
        (Assume single key joins)** * * **RULE:** The `ON` clause connects tables based *solely* on their defined key 
        relationship. Nothing else. * **ALL OTHER FILTERING BELONGS IN THE `WHERE` CLAUSE.** Apply `WHERE` conditions 
        *after* all joins are complete.
        
        4.  **`INNER JOIN` Usage:** Use `INNER JOIN` ONLY if the request explicitly requires results where matches 
        MUST exist in ALL joined tables.
        
        5.  **Schema Adherence:** Use ONLY the tables and columns provided in the schema section above. Do not invent 
        tables or columns.
        
        **=== OTHER QUERY RULES ===**
        - Use exact schema names. Use aliases (e.g., `d` for `departments`). Qualify columns (`d.name`).
        - Handle potential division by zero: `NULLIF(denominator, 0)`.
        - Quote identifiers only if needed (`"Table Name"."Column"`).
        - Use `ORDER BY` if sorting is implied.
        - Be careful with `WHERE` on the right side of `LEFT JOIN` (can act like `INNER JOIN`).
        
        **=== TASK ===**
        Task Description: {{{{TASK_DESCRIPTION_PLACEHOLDER}}}}
        Required Data Summary: {{{{REQUIRED_DATA_PLACEHOLDER}}}}
        
        **=== OUTPUT FORMAT ===**
        - Raw SQL query ONLY.
        - No explanations, comments, markdown (like ```sql).
        """
        # --- END MODIFIED SQL INSTRUCTION ---

        try:
            sql_gemini = initialize_gemini_model(system_instruction=sql_instruction, model_name="gemini-1.5-flash")
            logger.debug("SQL Gemini model initialized.")
        except Exception as model_init_error:
            logger.error(f"❌ Failed to initialize SQL Gemini model: {model_init_error}", exc_info=True)
            return [{"type": "text",
                     "data": f"Error: Failed to initialize the SQL generation component. {model_init_error}"}]

        for i, task in enumerate(tasks):
            task_description = task.get('description', f'Unnamed Task {i + 1}')
            task_type = task.get('task_type')
            required_data = task.get('required_data_summary', 'No data summary provided')
            viz_type = task.get('visualization_type')  # Will be None if not visualization

            logger.info(f"\n  Task {i + 1}/{len(tasks)}: '{task_description}' ({task_type})")

            if not task_type or not required_data:
                logger.warning(f"  [!] Skipping task - missing type or data summary")
                results.append(
                    {"type": "text",
                     "data": f"Skipping sub-task '{task_description}' due to incomplete definition from AI."})
                continue

            try:
                # --- Step 3a: Generate SQL Query using AI ---
                logger.info(f"    Generating SQL...")
                if sql_gemini is None:
                    logger.error("   [✗] SQL Gemini model not initialized!")
                    results.append({"type": "text",
                                    "data": f"Error processing task '{task_description}': SQL generation component "
                                            f"not ready."})
                    continue

                sql_prompt = (f"Task Description: {task_description}\nRequired Data Summary: {required_data}\nGenerate "
                              f"the PostgreSQL query using ONLY the provided schema and adhering strictly to the JOIN "
                              f"strategy and other rules.")

                sql_chat = sql_gemini.start_chat()
                sql_response = sql_chat.send_message(sql_prompt)
                sql_query_text = clean_response_text(sql_response.text)
                logger.debug(f"Generated SQL: {sql_query_text}")

                if not sql_query_text or not sql_query_text.lower().strip().startswith("select"):
                    logger.warning(f"    [✗] Invalid or empty SQL query generated by AI: '{sql_query_text}'")
                    results.append({"type": "text",
                                    "data": f"Could not generate a valid SQL query for task: '{task_description}'. AI "
                                            f"Output: '{sql_query_text}'"})
                    continue
                logger.info(f"    [✓] SQL query generated")

                # --- Step 3b: Fetch Data from Database ---
                logger.info(f"    Fetching data...")
                data = sql_query(sql_query_text)  # Returns list of dicts or empty list

                if not data:
                    logger.info(f"    [!] Query returned no data")
                    results.append({"type": "text",
                                    "data": f"For '{task_description}': The query executed successfully but returned "
                                            f"no data."})
                    continue
                else:
                    logger.info(f"    [✓] Fetched {len(data)} records")

                # --- Step 3c: Generate Insight, Visualization, or Report ---
                # (Insight Generation Logic)
                if task_type == 'insight':
                    logger.info(f"    Generating insight...")
                    if insight_gemini is None:
                        insight_instruction = """You are an analyst. Based on the provided data (in JSON format) and 
                        the original request, generate a concise textual insight. - Focus on answering the specific 
                        question asked in the 'Original Request'. - Be factual and base your answer ONLY on the 
                        provided data. - Keep the insight brief (1-3 sentences). - Output ONLY the insight text. No 
                        extra formatting or greetings."""
                        try:
                            insight_gemini = initialize_gemini_model(model_name="gemini-1.5-flash",
                                                                     system_instruction=insight_instruction)
                            logger.debug("Insight Gemini model initialized.")
                        except Exception as model_init_error:
                            logger.error(f"   [✗] Failed to initialize Insight Gemini model: {model_init_error}",
                                         exc_info=True)
                            results.append({"type": "text",
                                            "data": f"Error processing task '{task_description}': Insight generation "
                                                    f"component failed to initialize."})
                            continue

                    insight_prompt = f"""
                    Data (JSON format):
                    {json.dumps(data, indent=2, default=str)}
                    
                    Original Request for this Insight:
                    "{task_description}"
                    
                    Generate the insight based *only* on the data provided:
                    """
                    insight_chat = insight_gemini.start_chat()
                    insight_response = insight_chat.send_message(insight_prompt)
                    insight_text = clean_response_text(insight_response.text)
                    logger.debug(f"Generated Insight: {insight_text}")
                    logger.info(f"    [✓] Insight generated")
                    results.append({"type": "text", "data": insight_text})

                # (Visualization Generation Logic)
                elif task_type == 'visualization':
                    viz_type_str = viz_type if viz_type else "chart"
                    logger.info(f"    Generating {viz_type_str} visualization...")

                    if not viz_type or viz_type not in ['bar', 'line']:
                        logger.warning(
                            f"    [!] Invalid or missing visualization type '{viz_type}' specified for task.")
                        results.append({"type": "text",
                                        "data": f"Skipping visualization for '{task_description}': Invalid or missing "
                                                f"chart type ('{viz_type}'). Requires 'bar' or 'line'."})
                        continue

                    if plotly_gemini is None:
                        plotly_instruction = f""" You are a data visualization expert using Plotly.js. Given a 
                        dataset (as a JSON list of objects), a description of the desired visualization, 
                        and the required chart type (bar or line), generate the Plotly JSON configuration (
                        specifically the 'data' and 'layout' objects).
                        
                        Rules: - Create a meaningful title for the chart based on the description. Use the exact 
                        column names from the dataset for 'x' and 'y' keys in the data trace. - Ensure the generated 
                        JSON is syntactically correct and contains ONLY the 'data' (list) and 'layout' (object) keys 
                        at the top level. - Map the data fields appropriately to x and y axes based on the 
                        description and chart type ('bar' or 'line'). Infer appropriate axes from the data keys. - 
                        Generate ALL necessary fields for a basic, valid Plotly chart (e.g., 'type', 'x', 
                        'y' in trace; 'title' in layout). - ONLY output the JSON object starting with `{{` and ending 
                        with `}}`. - Do not include any explanations, comments, code blocks (like ```json), 
                        or other text.
                        
                        Example Output Format: {{ "data": [{{ "x": [/* array of x-values */], "y": [/* array of 
                        y-values */], "type": "bar", "name": "Optional Trace Name" }}], "layout": {{ "title": "Chart 
                        Title Based on Description", "xaxis": {{"title": "X-Axis Label"}}, "yaxis": {{"title": 
                        "Y-Axis Label"}} }} }}"""
                        try:
                            plotly_gemini = initialize_gemini_model(system_instruction=plotly_instruction)
                            logger.debug("Plotly Gemini model initialized.")
                        except Exception as model_init_error:
                            logger.error(f"   [✗] Failed to initialize Plotly Gemini model: {model_init_error}",
                                         exc_info=True)
                            results.append({"type": "text",
                                            "data": f"Error processing task '{task_description}': Visualization "
                                                    f"generation component failed to initialize."})
                            continue

                    data_keys = list(data[0].keys())  # data is guaranteed to exist here
                    plotly_prompt = f"""
                    Dataset (JSON format, keys available: {data_keys}):
                    {json.dumps(data, indent=2, default=str)}
                    
                    Visualization Description:
                    "{task_description}"
                    
                    Required Chart Type:
                    "{viz_type}"
                    
                    Generate the Plotly JSON configuration ('data' and 'layout' objects only):
                    """
                    plotly_chat = plotly_gemini.start_chat()
                    plotly_response = plotly_chat.send_message(plotly_prompt)
                    plotly_json_text = clean_response_text(plotly_response.text)
                    logger.debug(f"Raw Plotly JSON response: {plotly_response.text}")
                    logger.debug(f"Cleaned Plotly JSON response: {plotly_json_text}")

                    try:
                        if not plotly_json_text.startswith('{') or not plotly_json_text.endswith('}'):
                            raise ValueError("Plotly response is not a valid JSON object string.")
                        plotly_json = json.loads(plotly_json_text)
                        # Basic validation
                        if not isinstance(plotly_json,
                                          dict) or 'data' not in plotly_json or 'layout' not in plotly_json:
                            raise ValueError(
                                "Plotly JSON missing 'data' or 'layout' key at the top level, or is not an object.")
                        if not isinstance(plotly_json['data'], list):
                            raise ValueError("Plotly 'data' key must be a list.")
                        if not isinstance(plotly_json['layout'], dict):
                            raise ValueError("Plotly 'layout' key must be an object.")

                        logger.info(f"    [✓] Visualization ({viz_type}) generated")
                        results.append({"type": "graph", "data": plotly_json})
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.error(f"    [✗] Failed to parse or validate Plotly JSON: {e}", exc_info=True)
                        logger.error(f"    Problematic Plotly JSON text: {plotly_json_text}")
                        results.append({"type": "text",
                                        "data": f"Error generating visualization for '{task_description}': Invalid "
                                                f"Plotly configuration received from AI. Details: {e}"})

                # (Report Generation Logic)
                elif task_type == 'report':
                    logger.info(f"    Generating report data...")
                    try:
                        columns = list(data[0].keys())
                        report_data = {
                            "columns": columns,
                            "rows": data
                        }
                        logger.info(f"    [✓] Report data prepared")
                        results.append({"type": "table", "data": report_data})  # Use "table" as the type
                    except Exception as report_err:
                        logger.error(f"    [✗] Failed to format data for report: {report_err}", exc_info=True)
                        results.append({"type": "text",
                                        "data": f"Error preparing report data for '{task_description}': {report_err}"})

                else:
                    logger.warning(f"    [!] Unknown task type '{task_type}'")
                    results.append({"type": "text",
                                    "data": f"Unknown task type '{task_type}' encountered for sub-task"
                                            f" '{task_description}'"
                                            f". Cannot process."})

            except Exception as task_error:
                logger.error(f"    [✗] Error processing task '{task_description}': {task_error}",
                             exc_info=True)  # Keep exc_info for task errors
                results.append({"type": "text",
                                "data": f"An error occurred while processing "
                                        f"sub-task '{task_description}': {task_error}"})
            # End of task processing try-except block
        # End of loop through tasks

    except Exception as e:
        logger.error(f"❌ An unexpected error occurred during the main processing: {e}", exc_info=True)
        results.append({"type": "text", "data": f"An unexpected error occurred: {e}"})
    # End of main try-except block

    logger.info("\n🏁 PIPELINE EXECUTION COMPLETE")
    if results:
        logger.info(f"Returning {len(results)} result items.")
    else:
        logger.info("No results were generated.")

    return results
