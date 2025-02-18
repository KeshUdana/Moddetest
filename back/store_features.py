import requests
from flask_pymongo import PyMongo
import tensorflow as tf
import numpy as np
from PIL import Image
import io
from app import app  # Assuming app.py is in the same directory

mongo = PyMongo(app)

# Load pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

def preprocess_image(image_data):
    img = Image.open(io.BytesIO(image_data))
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def extract_features(image_data):
    img_array = preprocess_image(image_data)
    features = model.predict(img_array)
    return features.flatten()

def store_product_features():
    products = [
        {'name': 'T-shirt', 'image_url': 'https://example.com/tshirt.jpg', 'price': 25.99, 'description': 'A comfortable t-shirt'},
        {'name': 'Jeans', 'image_url': 'https://example.com/jeans.jpg', 'price': 49.99, 'description': 'Stylish denim jeans'}
    ]
    
    for product in products:
        image_data = requests.get(product['image_url']).content
        features = extract_features(image_data)
        product['features'] = features
        mongo.db.products.insert_one(product)

if __name__ == '__main__':
    store_product_features()
