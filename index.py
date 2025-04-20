from flask import Flask, request, jsonify
from rag_pipeline import process_prompt

app = Flask(__name__)


@app.route('/api/query', methods=['POST'])
def query_excel():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({"error": "Missing 'prompt' in request body"}), 400

    try:
        result = process_prompt(prompt)
        return jsonify({"response": result}), 200
    except Exception as e:
        return jsonify({"error": f"Error processing prompt: {str(e)}"}), 500


@app.route('/', methods=['GET'])
def home():
    return jsonify({"status": "API is running", "endpoints": ["/api/query"]}), 200


if __name__ == '__main__':
    app.run()
