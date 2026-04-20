#FastAPI wrapper around the ResNet-20 CIFAR-10 classifier.
#Loads the best checkpoint once at startup, classifies uploaded images.
#Run it:
#   uvicorn api.app:app --reload
#   (or pick the API option from main.py)
import os
import sys
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import torch
from PIL import Image
from torchvision import transforms

from model import resnet20
from train import BEST_PATH, pick_device
from data import CLASSES, CIFAR10_MEAN, CIFAR10_STD


app = FastAPI(
    title="CIFAR-10 Image Classifier API",
    description="ResNet-20 trained from scratch on CIFAR-10. Upload any image.",
    version="1.0.0",
)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

#Global model -- loaded once at startup.
model = None
device = None
transform = None


@app.on_event("startup")
def load_on_startup():
    global model, device, transform
    print("Loading ResNet-20 checkpoint...")
    device = pick_device()

    if not os.path.exists(BEST_PATH):
        print("WARNING: No checkpoint found at " + BEST_PATH + ". Train first.")
        return

    ckpt = torch.load(BEST_PATH, map_location=device)
    model = resnet20(num_classes=10).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    #Same preprocessing as evaluation: resize to 32x32, normalize.
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    print("Model ready (epoch " + str(ckpt.get("epoch", "?")) + ").")


@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    #Read and validate the uploaded image.
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file as an image.")

    #Preprocess and run inference.
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]

    top5_probs, top5_indices = torch.topk(probs, 5)
    predictions = []
    for i in range(5):
        idx = int(top5_indices[i].item())
        predictions.append({
            "class": CLASSES[idx],
            "probability": round(float(top5_probs[i].item()), 4),
        })

    top = predictions[0]
    return {
        "label": top["class"],
        "confidence": top["probability"],
        "top5": predictions,
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}
