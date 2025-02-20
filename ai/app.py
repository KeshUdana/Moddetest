from flask import Flask, request, jsonify
import requests

app = Flask(__name__)

FEATURE_EXTRACTION_SERVICE_URL = "http://127.0.0.1:5001/extract_features"

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    files = {'image': file}
    response = requests.post(FEATURE_EXTRACTION_SERVICE_URL, files=files)

    if response.status_code != 200:
        return jsonify({'error': 'Feature extraction failed', 'details': response.json()}), 500

    extracted_features = response.json().get("features", [])

    return jsonify({'extracted_features': extracted_features})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
