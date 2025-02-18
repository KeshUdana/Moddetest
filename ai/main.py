from fastapi import FastAPI, File, UploadFile
import numpy as np
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input

from fastapi.responses import JSONResponse
import uvicorn

app = FastAPI()

# Load MobileNet model (pre-trained)
model = MobileNet(weights='imagenet', include_top=False, pooling='avg')

def extract_features(img_bytes):
    # Load and preprocess the image
    img = image.load_img(img_bytes, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)

    # Extract features
    features = model.predict(img_data)
    return features.flatten()

@app.post("/extract-features")
async def extract_features_api(file: UploadFile = File(...)):
    try:
        # Read image bytes
        img_bytes = image.load_img(file.file, target_size=(224, 224))

        # Extract features
        features = extract_features(img_bytes)

        return JSONResponse(content={"features": features.tolist()})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
