import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from gemini_fallback import get_precautions_and_water_level

# Set image path (you can change this)
image_path = r"C:\Users\HP\OneDrive\Desktop\AI Plants\Backend\Data\PlantVillage\Potato___healthy\00fc2ee5-729f-4757-8aeb-65c3355874f2___RS_HL 1864.JPG"

# Load your trained model
model = load_model("Notebook/plant_disease_final_model.h5")

# Load class names
with open("Notebook/class_names.json", "r") as f:
    class_names = json.load(f)

# Check if the image exists
if not os.path.exists(image_path):
    print("❌ Image file does not exist!")
    exit()

# Preprocess the image
img = image.load_img(image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Predict the class
prediction = model.predict(img_array)
predicted_index = np.argmax(prediction)
predicted_class = class_names[predicted_index]

print(f"\n✅ Predicted Disease: {predicted_class}")

# Get Gemini response
response = get_precautions_and_water_level(predicted_class)

print("\n🌿 Gemini Recommendations:")
print(f"🛡️  Precautions: {response.get('precautions', 'N/A')}")
print(f"💧 Water Level: {response.get('water_level', 'N/A')}")
