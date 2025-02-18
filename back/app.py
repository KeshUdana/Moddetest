from flask import Flask, request, jsonify
from flask_pymongo import PyMongo
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import requests

# Initialize Flask app and MongoDB connection
app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/your_db_name"  # Update with your MongoDB URI
mongo = PyMongo(app)

# Load pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False  # Freezing the layers as we only need feature extraction

# Utility function to preprocess images for MobileNetV2
def preprocess_image(image_data):
    img = Image.open(io.BytesIO(image_data))
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Feature extraction function using MobileNetV2
def extract_features(image_data):
    img_array = preprocess_image(image_data)
    features = model.predict(img_array)  # Extract features
    return features.flatten()  # Flatten the features array

# Function to calculate cosine similarity
def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    return dot_product / (magnitude_a * magnitude_b)

# Endpoint to receive image and find similar products
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Extract features from the uploaded image
    image_data = file.read()
    user_features = extract_features(image_data)

    # Compare with products in the MongoDB database
    similar_products = []
    products = mongo.db.products.find()  # Fetch all products from MongoDB

    for product in products:
        image_url = product.get('image_url')
        product_features = product.get('features')  # Pre-stored features in MongoDB
        if product_features:
            similarity = cosine_similarity(user_features, product_features)
            similar_products.append({
                'name': product.get('name'),
                'image_url': image_url,
                'price': product.get('price'),
                'description': product.get('description'),
                'similarity': similarity
            })

    # Sort products by similarity and return top 5
    similar_products = sorted(similar_products, key=lambda x: x['similarity'], reverse=True)[:5]
    return jsonify(similar_products)

if __name__ == '__main__':
    app.run(debug=True)
