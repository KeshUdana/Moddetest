from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import tensorflow as tf
import os

app = Flask(__name__)

# Load Pre-trained MobileNetV2 Model (Global)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D()  # Flatten the features
])

# Extract Features from a Single Image
def extract_features(img_path):
    img = Image.open(img_path)
    img = img.resize((224, 224))
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img = np.array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32')
    img = img / 255.0
    features = model.predict(img)
    return features.flatten()

# Extract Features from Dataset Images
def extract_features_from_dataset(dataset_dir):
    features_list = []
    image_paths = []
    for root, dirs, files in os.walk(dataset_dir):
        for img_name in files:
            if img_name.lower().endswith(('jpg', 'jpeg', 'png', 'bmp')):
                img_path = os.path.join(root, img_name)
                try:
                    features = extract_features(img_path)
                    features_list.append(features)
                    image_paths.append(img_path)
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
    return np.array(features_list), image_paths

# Compute Similarity
def get_most_similar_images(user_img_path, dataset_dir, top_n=5):
    user_features = extract_features(user_img_path)
    dataset_features, image_paths = extract_features_from_dataset(dataset_dir)
    similarities = cosine_similarity([user_features], dataset_features)[0]
    most_similar_idx = np.argsort(similarities)[-top_n:][::-1]
    similar_images = [image_paths[idx] for idx in most_similar_idx]
    return similar_images, similarities[most_similar_idx]

# Flask Endpoint
@app.route('/find-similar', methods=['POST'])
def find_similar():
    # Get the uploaded image and dataset directory from the request
    user_img = request.files['image']
    dataset_dir = request.form.get('dataset_dir', '/path/to/dataset')  # Default dataset path

    # Save the uploaded image temporarily
    user_img_path = f"/tmp/{user_img.filename}"
    user_img.save(user_img_path)

    # Get similar images
    similar_images, similarities = get_most_similar_images(user_img_path, dataset_dir, top_n=5)

    # Return the results as JSON
    return jsonify({
        "similar_images": similar_images,
        "similarities": similarities.tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)