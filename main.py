# from fastapi import FastAPI
# from pydantic import BaseModel
# import base64
# import io
# from PIL import Image
# import numpy as np
# from tensorflow.keras.models import load_model
# from gemini_fallback import get_precautions_and_water_level
# import json

# app = FastAPI()

# # Load your model and class names once on startup
# model = load_model("Notebook/plant_disease_final_model.h5")
# with open("Notebook/class_names.json", "r") as f:
#     class_names = json.load(f)

# class PredictRequest(BaseModel):
#     image_base64: str

# def preprocess_image(base64_str: str) -> np.ndarray:
#     img_data = base64.b64decode(base64_str)
#     img = Image.open(io.BytesIO(img_data)).convert("RGB")
#     img = img.resize((224, 224))
#     img_array = np.array(img) / 255.0
#     img_array = np.expand_dims(img_array, axis=0)
#     return img_array

# @app.post("/predict")
# async def predict(request: PredictRequest):
#     try:
#         img_array = preprocess_image(request.image_base64)
#         preds = model.predict(img_array)
#         predicted_index = np.argmax(preds[0])
#         predicted_class = class_names[predicted_index]

#         gemini_response = get_precautions_and_water_level(predicted_class)

#         return {
#             "disease": predicted_class,
#             "precautions": gemini_response.get("precautions", "N/A"),
#             "water_level": gemini_response.get("water_level", "N/A"),
#         }
#     except Exception as e:
#         return {"error": str(e)}

# # Add this to run the server
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI
from pydantic import BaseModel
import base64
import io
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from gemini_fallback import get_precautions_and_water_level
import json

app = FastAPI()

# Load model and classes once
model = load_model("Notebook/plant_disease_final_model.h5")
with open("Notebook/class_names.json", "r") as f:
    class_names = json.load(f)

class PredictRequest(BaseModel):
    image_base64: str

def preprocess_image(base64_str: str) -> np.ndarray:
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/api/predict")  # NOTE: prefix with /api
async def predict(request: PredictRequest):
    try:
        img_array = preprocess_image(request.image_base64)
        preds = model.predict(img_array)
        predicted_index = np.argmax(preds[0])
        predicted_class = class_names[predicted_index]

        gemini_response = get_precautions_and_water_level(predicted_class)

        return {
            "disease": predicted_class,
            "precautions": gemini_response.get("precautions", "N/A"),
            "water_level": gemini_response.get("water_level", "N/A"),
        }
    except Exception as e:
        return {"error": str(e)}
