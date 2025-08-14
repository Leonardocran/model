from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import numpy as np
import requests
from io import BytesIO
from PIL import Image
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image as keras_image

# Load model
model = VGG19(weights='imagenet')

app = Flask(__name__)
CORS(app)  # Allow all origins (for dev use)

def predict_from_url(url):
    try:
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            return {"error": f"Failed to download image (HTTP {response.status_code})"}
        
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = img.resize((224, 224))
        
        x = keras_image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        preds = model.predict(x)
        decoded = decode_predictions(preds, top=5)[0]
        
        return [
            {"id": imagenetID, "label": label, "confidence": float(score)}
            for (imagenetID, label, score) in decoded
        ]
    except Exception as e:
        return {"error": str(e)}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Please provide an image URL"}), 400
    
    url = data["url"]
    results = predict_from_url(url)
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
