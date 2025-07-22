# main.py
# To run this:
# 1. Install necessary libraries: pip install fastapi "uvicorn[standard]" python-multipart Pillow torch torchvision opencv-python numpy
# 2. Place your model file (e.g., 'final_model.pth') in the same directory.
# 3. Run the server from your terminal: uvicorn main:app --reload

import os
import cv2
import numpy as np
from PIL import Image, ImageChops
import torch
import torch.nn as nn
from torchvision import models, transforms
import torch.nn.functional as F
import io
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- 1. Configuration & Setup ---
DEVICE = torch.device("cpu")
# --- UPDATE THIS PATH to your model file ---
# For this example, we assume the model file is in the same directory
MODEL_PATH = 'final_model_lightweight.pth' 
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please place your trained .pth file here.")

# --- Class Mappings ---
idx_to_class = {0: 'fake', 1: 'real'}
class_to_idx = {v: k for k, v in idx_to_class.items()}

# --- FastAPI App Initialization ---
app = FastAPI(
    title="DeepReveal API",
    description="A backend API for the DeepReveal project to detect and localize deepfakes.",
    version="1.0.0"
)

# --- CORS (Cross-Origin Resource Sharing) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


# --- 2. Model Definition and Preprocessing Functions ---

def generate_ela(image, quality=90):
    output_io = io.BytesIO()
    image.save(output_io, "JPEG", quality=quality)
    output_io.seek(0)
    temp_image = Image.open(output_io)
    ela_image = ImageChops.difference(image, temp_image)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0: max_diff = 1
    scale = 255.0 / max_diff
    return Image.eval(ela_image, lambda p: p * scale)

def generate_residual(image):
    img_cv = np.array(image)
    denoised_img = cv2.medianBlur(img_cv, 3)
    residual = cv2.absdiff(img_cv, denoised_img)
    return Image.fromarray(residual)

def generate_fft(image):
    img_gray = image.convert('L')
    img_gray_np = np.array(img_gray)
    f = np.fft.fft2(img_gray_np)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return Image.fromarray(magnitude_spectrum).convert('L')

class DeepRevealModel4Branch(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.rgb_branch = models.efficientnet_b0(weights=None)
        self.ela_branch = models.efficientnet_b0(weights=None)
        self.res_branch = models.efficientnet_b0(weights=None)
        num_features = self.rgb_branch.classifier[1].in_features
        self.rgb_branch.classifier = nn.Identity()
        self.ela_branch.classifier = nn.Identity()
        self.res_branch.classifier = nn.Identity()
        self.fft_cnn = nn.Sequential(nn.Conv2d(1, 16, 3, 2, 1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Sequential(nn.Linear(num_features * 3 + 32, 512), nn.ReLU(), nn.Dropout(0.5), nn.Linear(512, num_classes))

    def forward(self, rgb, ela, res, fft):
        feat_rgb = self.rgb_branch(rgb)
        feat_ela = self.ela_branch(ela)
        feat_res = self.res_branch(res)
        feat_fft = self.fft_cnn(fft).flatten(1)
        fused = torch.cat((feat_rgb, feat_ela, feat_res, feat_fft), dim=1)
        return self.classifier(fused)

# --- Load Model ---
model = DeepRevealModel4Branch().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("--- DeepReveal Model Loaded Successfully ---")


# --- 3. Prediction Logic with Grad-CAM (Robust Fix) ---

def get_prediction(input_image: Image.Image):
    """
    Takes a PIL image, performs prediction and Grad-CAM, 
    and returns results and the result image as a base64 string.
    """
    original_img_pil = input_image.convert('RGB')
    
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    fft_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    ela_img = generate_ela(original_img_pil)
    res_img = generate_residual(original_img_pil)
    fft_img = generate_fft(original_img_pil)
    
    rgb_tensor = data_transform(original_img_pil).unsqueeze(0).to(DEVICE)
    ela_tensor = data_transform(ela_img).unsqueeze(0).to(DEVICE)
    res_tensor = data_transform(res_img).unsqueeze(0).to(DEVICE)
    fft_tensor = fft_transform(fft_img).unsqueeze(0).to(DEVICE)

    rgb_tensor.requires_grad = True

    output = model(rgb_tensor, ela_tensor, res_tensor, fft_tensor)
    probabilities = F.softmax(output, dim=1)[0]
    prediction_idx = probabilities.argmax().item()
    prediction_label = idx_to_class[prediction_idx].upper()
    confidences = {idx_to_class[i].upper(): f"{prob.item():.4f}" for i, prob in enumerate(probabilities)}

    img_to_draw_on = np.array(original_img_pil.resize((224, 224)))

    if prediction_label == 'FAKE':
        target_layer = model.rgb_branch.features[-1]
        gradients, features = [], []
        
        def backward_hook(module, grad_input, grad_output): gradients.append(grad_output[0])
        def forward_hook(module, input, output): features.append(output)
            
        backward_handle = target_layer.register_backward_hook(backward_hook)
        forward_handle = target_layer.register_forward_hook(forward_hook)

        score = output[:, class_to_idx['fake']]
        model.zero_grad()
        score.backward()
        
        # Clean up hooks immediately
        backward_handle.remove()
        forward_handle.remove()

        # --- ROBUSTNESS FIX ---
        # Defensively check if gradients were captured before proceeding.
        if gradients and features:
            try:
                grads = gradients[0].cpu().data.numpy()
                fmap = features[0].cpu().data.numpy()
                
                weights = np.mean(grads, axis=(2, 3))[0, :]
                cam = np.zeros(fmap.shape[2:], dtype=np.float32)
                for i, w in enumerate(weights):
                    cam += w * fmap[0, i, :, :]
                    
                cam = np.maximum(cam, 0)
                cam = cv2.resize(cam, (224, 224))
                cam -= np.min(cam)
                if np.max(cam) > 0: cam /= np.max(cam)

                heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                overlay = cv2.addWeighted(img_to_draw_on, 0.6, heatmap, 0.4, 0)
                img_to_draw_on = overlay
            except Exception as e:
                print(f"Could not generate Grad-CAM heatmap due to an error: {e}")
        else:
            print("Warning: Could not generate Grad-CAM heatmap. Gradients were not captured.")
    
    color = (0, 0, 255) if "FAKE" in prediction_label else (0, 255, 0)
    conf_text = f"Prediction: {prediction_label}"
    cv2.putText(img_to_draw_on, conf_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Convert final image to base64 to send via JSON
    result_pil = Image.fromarray(img_to_draw_on)
    buff = io.BytesIO()
    result_pil.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")

    return prediction_label, confidences, f"data:image/png;base64,{img_str}"


# --- 4. API Endpoints ---

class PredictionResponse(BaseModel):
    prediction: str
    confidence: dict
    result_image: str # Base64 encoded image string

@app.get("/")
def read_root():
    """A simple endpoint to check if the API is running."""
    return {"message": "Welcome to the DeepReveal API. Use the /predict/ endpoint to analyze an image."}

@app.post("/predict/", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """
    Receives an image file, processes it, and returns the prediction and result image.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        
        prediction, confidence, result_image_str = get_prediction(pil_image)
        
        return {
            "prediction": prediction,
            "confidence": confidence,
            "result_image": result_image_str
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")
