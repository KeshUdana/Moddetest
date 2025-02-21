import tensorflow as tf

# Load Pre-trained MobileNetV2 Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D()  # Flatten the features
])
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
from PIL import Image

def preprocess_input_image(img_path):
    img = Image.open(img_path)  # Open the image from the provided path
    img = img.resize((224, 224))  # Resize image to match MobileNetV2 input
    img = np.array(img)
    if img.shape[2] == 4:  # Handle images with transparency (alpha channel)
        img = img[:, :, :3]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = preprocess_input(img)  # Preprocess for MobileNetV2 (scaling pixel values)
    return img
def extract_features(img_path):
    img = Image.open(img_path)  # Open the image from the provided path
    img = img.resize((224, 224))  # Resize image to match MobileNetV2 input size

    # Check if the image has an alpha channel (RGBA), and remove it if present
    if img.mode == 'RGBA':
        img = img.convert('RGB')  # Convert to RGB by discarding the alpha channel

    img = np.array(img)  # Convert the image to a NumPy array
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img.astype('float32')  # Convert the image to float32 type
    img = img / 255.0  # Normalize the image

    # Extract features using MobileNetV2 model
    features = model.predict(img)
    return features.flatten()  # Flatten to make it 1D vector

import os

def extract_features_from_dataset(dataset_dir):
    features_list = []
    image_paths = []
    
    # Use os.walk to handle subdirectories and files
    for root, dirs, files in os.walk(dataset_dir):  # os.walk handles subdirectories
        for img_name in files:
            # Only process image files (check if it's a valid image extension)
            if img_name.lower().endswith(('jpg', 'jpeg', 'png', 'bmp')):
                img_path = os.path.join(root, img_name)
                try:
                    features = extract_features(img_path)
                    features_list.append(features)
                    image_paths.append(img_path)
                except Exception as e:
                    print(f"Error processing image {img_path}: {e}")
    
    return np.array(features_list), image_paths

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_most_similar_images(user_img_path, dataset_dir, top_n=5):
    # Step 1: Extract features for the user image
    user_features = extract_features(user_img_path)

    # Step 2: Extract features from the dataset
    dataset_features, image_paths = extract_features_from_dataset(dataset_dir)

    # Step 3: Calculate similarity between user image and dataset images using cosine similarity
    similarities = cosine_similarity([user_features], dataset_features)[0]

    # Step 4: Get top N most similar images
    most_similar_idx = np.argsort(similarities)[-top_n:][::-1]
    similar_images = [image_paths[idx] for idx in most_similar_idx]

    return similar_images, similarities[most_similar_idx]

import matplotlib.pyplot as plt
from PIL import Image

def display_similar_images(user_img_path, similar_images, similarities):
    # Display the user image
    img = Image.open(user_img_path)
    plt.figure(figsize=(12, 12))
    plt.subplot(3, 3, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("User Image")

    # Display the most similar images
    for i, (sim_img, similarity) in enumerate(zip(similar_images, similarities)):
        img = Image.open(sim_img)
        plt.subplot(3, 3, i+2)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Similarity: {similarity:.2f}")

    plt.show()

# Example usage:
user_img_path = '/kaggle/input/testing-imageskeshawa/womanSkirt.png'  # Path to the user's uploaded image
dataset_dir = '/kaggle/input/moddelite/ModdeDataset/skirt'  # Path to your fashion dataset

# Step 1: Get the most similar images
similar_images, similarities = get_most_similar_images(user_img_path, dataset_dir, top_n=5)

# Step 2: Display the results
display_similar_images(user_img_path, similar_images, similarities)