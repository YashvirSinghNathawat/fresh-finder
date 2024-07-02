from pathlib import Path
from fastapi import UploadFile
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve(strict=True).parent

classes = {
    0: 'fresh_apple',
    1: 'fresh_banana',
    2: 'fresh_bitter_gourd',
    3: 'fresh_capsicum',
    4: 'fresh_orange',
    5: 'fresh_tomato',
    6: 'stale_apple',
    7: 'stale_banana',
    8: 'stale_bitter_gourd',
    9: 'stale_capsicum',
    10: 'stale_orange',
    11: 'stale_tomato',
}


model = load_model(f"{BASE_DIR}/model.keras")

def preprocess_image(image: Image.Image,target_size=(256, 256)):
    image = image.convert('RGB')
    image = image.resize((256, 256))  # Resize to target size
    image_array = np.array(image)/255.0
    image_array = np.expand_dims(image_array,axis = 0)
    return image_array

def predict_pipeline(file: UploadFile):  # we can use bytes but its whole content is stored in memory
    # UploadFile provides the uploaded file as a stream, and Image.
    # open can read this stream and convert it into an image object that can be manipulated and processed.
    image = Image.open(file.file)
    image_array = preprocess_image(image)
    pred = model.predict(image_array)
    pred_class = classes[np.argmax(pred)]
    return pred_class