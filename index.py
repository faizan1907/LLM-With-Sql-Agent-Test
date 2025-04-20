import os
import logging
from flask import Flask, request, jsonify
from rag_pipeline import process_prompt

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

@app.route('/api/query', methods=['POST'])
def query_excel():
    logger.info("Received request to /api/query")
    if not request.is_json:
        logger.warning("Request is not JSON")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    prompt = data.get('prompt')

    if not prompt:
        logger.warning("Missing prompt in request")
        return jsonify({"error": "Missing 'prompt' in request body"}), 400

    try:
        logger.info(f"Processing prompt: {prompt[:50]}...")
        result = process_prompt(prompt)
        logger.info("Successfully processed prompt")
        return jsonify({"response": result}), 200
    except Exception as e:
        logger.error(f"Error processing prompt: {str(e)}", exc_info=True)
        return jsonify({"error": f"Error processing prompt: {str(e)}"}), 500

@app.route('/', methods=['GET'])
def home():
    logger.info("Home endpoint accessed")
    return jsonify({"status": "API is running", "endpoints": ["/api/query"]}), 200

@app.route('/health', methods=['GET'])
def health():
    return "OK", 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)