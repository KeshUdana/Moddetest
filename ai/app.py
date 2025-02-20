from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

MODEL_PATH = "fine_tuned_model.h5"

if os.path.exists(MODEL_PATH):
    print("Loading fine-tuned model...")
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print("Loading base MobileNetV2 model...")
    base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False  

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax') 
    ])

def extract_features(image_data):
    img = Image.open(io.BytesIO(image_data))
    img = img.resize((224, 224))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    features = model.predict(img_array)
    return features.flatten().tolist()

@app.route('/extract_features', methods=['POST'])
def extract_features_api():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = request.files['image']
    image_data = image.read()

    try:
        features = extract_features(image_data)
        return jsonify({"features": features})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/update_model', methods=['POST'])
def update_model():
    if 'model' not in request.files:
        return jsonify({"error": "No model file provided"}), 400

    model_file = request.files['model']
    model_file.save(MODEL_PATH)

    global model
    model = tf.keras.models.load_model(MODEL_PATH)

    return jsonify({"message": "Model updated successfully!"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)