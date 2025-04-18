from flask import Flask, request, jsonify
from rag_pipeline import process_prompt, cleanup
import atexit

app = Flask(__name__)


@app.route('/query', methods=['POST'])
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


# Cleanup database connections on shutdown
atexit.register(cleanup)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=False)
