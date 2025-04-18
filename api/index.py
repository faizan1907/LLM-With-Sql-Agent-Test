# api/index.py (for Vercel)
from flask import Flask, request, jsonify
from rag_pipeline import process_prompt
from flask import Request

app = Flask(__name__)


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


# For local development only
if __name__ == '__main__':
    app.run()


# No atexit.register needed for Vercel

# Handler for Vercel
def handler(request: Request):
    return app(request)
