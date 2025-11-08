from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import uuid
import os
from PIL import Image

# ✅ Load model once at startup
model = keras.models.load_model("models/plant_disease_recog_model_pwp.keras", compile=False)
print("✅ Model loaded successfully.")

# ✅ Load class labels
with open("plant_disease.json", "r") as file:
    plant_disease = json.load(file)

# ✅ Initialize FastAPI app
app = FastAPI(title="Plant Disease Detection API")

# ✅ Create upload folder
os.makedirs("uploadimages", exist_ok=True)
app.mount("/uploadimages", StaticFiles(directory="uploadimages"), name="uploadimages")

# ✅ Helper functions
def extract_features(image_path):
    image = Image.open(image_path).resize((160, 160))
    feature = np.array(image)
    feature = np.expand_dims(feature, axis=0)
    return feature

def model_predict(image_path):
    img = extract_features(image_path)
    prediction = model.predict(img)
    prediction_label = plant_disease[np.argmax(prediction)]
    return prediction_label

# ✅ API endpoint for prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded file temporarily
    file_id = f"temp_{uuid.uuid4().hex}_{file.filename}"
    file_path = os.path.join("uploadimages", file_id)
    
    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Run prediction
    prediction = model_predict(file_path)

    # Return JSON response
    return JSONResponse({
        "prediction": prediction,
        "image_path": f"/uploadimages/{file_id}"
    })

# ✅ Root endpoint
@app.get("/")
async def root():
    return {"message": "🌿 Plant Disease Detection API is running! Upload an image via /docs."}
