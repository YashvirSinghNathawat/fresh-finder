from fastapi import FastAPI, UploadFile, File
from app.model.model import predict_pipeline
from fastapi.responses import HTMLResponse
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
# CORS middleware to allow all origins (replace with specific domains in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve(strict=True).parent

@app.get('/',response_class=HTMLResponse)
def main():
    with open(f"app/main.html","r") as file:
        html_content = file.read()
    return html_content

@app.post('/predict')
def predict(image_file: UploadFile = File(...)):
    res = predict_pipeline(image_file)
    return {"prediction":res}