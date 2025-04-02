import os
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
import cv2
import base64
import uvicorn

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

CLASS_LABELS = {0: 'No Tumor', 1: 'Positive Tumor'}

model_path = "model.keras"
weights_path = "model.weights.h5"
json_path = "model.json"

if os.path.exists(model_path):
    loaded_model = tf.keras.models.load_model(model_path)
elif os.path.exists(json_path) and os.path.exists(weights_path):
    with open(json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_path)
else:
    raise RuntimeError("No valid model file found. Ensure 'model.keras' or 'model.json' with 'model.weights.h5' exists.")

# Helper function to decode a base64 image
def get_cv2_image_from_base64_string(b64str):
    try:
        if "," in b64str:
            b64str = b64str.split(",")[1]
        nparr = np.frombuffer(base64.b64decode(b64str), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Decoded image is None")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

@app.get("/predict")
async def predict(image_base64: str):
    try:
        image = get_cv2_image_from_base64_string(image_base64)
        image = cv2.resize(image, (224, 224))
        image = np.expand_dims(image, axis=0) / 255.0

        prediction = loaded_model.predict(image)
        result = np.argmax(prediction, axis=1)[0]
        label = CLASS_LABELS[result]

        return {
            "prediction": prediction.tolist(),
            "class_index": int(result),
            "class_label": label
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error in prediction: {str(e)}")

@app.get("/")
async def root():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)