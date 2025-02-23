from fastapi import FastAPI, File, UploadFile
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
import tensorflow as tf
import os
from pymongo import MongoClient
from typing import List

app = FastAPI()

# Load the model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D()
])

# Connect to MongoDB
client = MongoClient("mongodb://username:password@host:port")
db = client["feature_db"]
collection = db["features"]

# Feature extraction functions
def preprocess_input_image(img_path):
    img = Image.open(img_path)
    img = img.resize((224, 224))
    img = np.array(img)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

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

def get_most_similar_images(user_features, dataset_features, image_paths, top_n=5):
    similarities = cosine_similarity([user_features], dataset_features)[0]
    most_similar_idx = np.argsort(similarities)[-top_n:][::-1]
    similar_images = [image_paths[idx] for idx in most_similar_idx]
    return similar_images, similarities[most_similar_idx]

# API Endpoints
@app.post("/extract_features")
async def extract_features_endpoint(file: UploadFile = File(...)):
    # Save the uploaded file temporarily
    with open("temp_image.jpg", "wb") as buffer:
        buffer.write(await file.read())
    
    # Extract features
    user_features = extract_features("temp_image.jpg")
    
    # Fetch features from MongoDB
    dataset_features = []
    image_paths = []
    for record in collection.find():
        dataset_features.append(np.array(record["features"]))
        image_paths.append(record["image_path"])
    
    # Compute similarity
    similar_images, similarities = get_most_similar_images(user_features, dataset_features, image_paths)
    
    return {"similar_images": similar_images, "similarities": similarities.tolist()}

@app.post("/upload_dataset")
async def upload_dataset(dataset_dir: str):
    # Extract features from the dataset
    features_list, image_paths = extract_features_from_dataset(dataset_dir)
    
    # Store features in MongoDB
    for features, img_path in zip(features_list, image_paths):
        collection.insert_one({"image_path": img_path, "features": features.tolist()})
    
    return {"message": "Dataset features uploaded successfully"}