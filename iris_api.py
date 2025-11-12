from flask import Flask, request, jsonify
import logging
import google.cloud.logging
from google.cloud.logging.handlers import CloudLoggingHandler

app = Flask(__name__)

# Initialize Google Cloud Logging
client = google.cloud.logging.Client()
client.setup_logging()

@app.route('/health')
def health():
    return jsonify({"status": "ok"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Dummy example
    sepal_length = data.get('sepal_length', 0)
    if sepal_length > 5:
        prediction = "versicolor"
    else:
        prediction = "setosa"

    app.logger.info(f"Prediction: {prediction} for input: {data}")
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
