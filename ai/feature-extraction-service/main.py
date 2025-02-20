from fastapi import FastAPI, File, UploadFile, HTTPException
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = FastAPI()

MODEL_PATH = "fine_tuned_model.h5"

# Load the model
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

# Extract features from an image
def extract_features(image_data: bytes) -> list:
    img = Image.open(io.BytesIO(image_data))
    img = img.resize((224, 224))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    features = model.predict(img_array)
    return features.flatten().tolist()

# Endpoint for feature extraction
@app.post("/extract_features")
async def extract_features_api(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_data = await file.read()
    try:
        features = extract_features(image_data)
        return {"features": features}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for updating the model
@app.post("/update_model")
async def update_model(file: UploadFile = File(...)):
    if not file.filename.endswith('.h5'):
        raise HTTPException(status_code=400, detail="File must be a .h5 model")

    model_data = await file.read()
    with open(MODEL_PATH, "wb") as f:
        f.write(model_data)

    global model
    model = tf.keras.models.load_model(MODEL_PATH)

    return {"message": "Model updated successfully!"}