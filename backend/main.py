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
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import firebase_admin
from firebase_admin import credentials, auth
import warnings

warnings.filterwarnings('ignore')

# --- 1. Configuration & Setup ---
DEVICE = torch.device("cpu") # Using CPU for broader compatibility
MODEL_PATH = 'final_model_lightweight.pth' # Ensure this path is correct

if not os.path.exists(MODEL_PATH):
    # For local testing, you might need to manually place this file.
    # In a deployment environment (like Docker), ensure it's copied.
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}.")

# --- Firebase Initialization ---
SERVICE_ACCOUNT_KEY_PATH = 'firebase-service-account.json' # Ensure this path is correct
if not os.path.exists(SERVICE_ACCOUNT_KEY_PATH):
    raise FileNotFoundError(f"Firebase service account key not found at {SERVICE_ACCOUNT_KEY_PATH}")

cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="DeepReveal API",
    description="A backend API for the DeepReveal project to detect and localize deepfakes.",
    version="1.5.2" # Version updated for both CAM fixes
)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # WARNING: For production, replace "*" with specific client origins (e.g., ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Authentication Dependency ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid authentication credentials: {e}",
            headers={"WWW-Authenticate": "Bearer"},
        )

# --- 2. Model Preprocessing Functions ---
def generate_ela(image, quality=90):
    output_io = io.BytesIO()
    image.save(output_io, "JPEG", quality=quality)
    output_io.seek(0)
    temp_image = Image.open(output_io)
    ela_image = ImageChops.difference(image, temp_image)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale = 255.0 / max_diff if max_diff != 0 else 1
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
    # CORRECTED: Removed the extra 'fft' here
    fshift = np.fft.fftshift(f) 
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return Image.fromarray(magnitude_spectrum).convert('L')

# --- 3. Model Definition ---
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
        
        self.fft_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(num_features * 3 + 32, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, rgb, ela, res, fft):
        feat_rgb = self.rgb_branch(rgb)
        feat_ela = self.ela_branch(ela)
        feat_res = self.res_branch(res)
        feat_fft = self.fft_cnn(fft).flatten(1)
        
        fused = torch.cat((feat_rgb, feat_ela, feat_res, feat_fft), dim=1)
        
        return self.classifier(fused)

# Wrapper class for Grad-CAM
class DeepRevealWithCAM(nn.Module):
    def __init__(self, model):
        super(DeepRevealWithCAM, self).__init__()
        self.model = model
        self.rgb_features = self.model.rgb_branch.features
        self.rgb_avgpool = self.model.rgb_branch.avgpool
        self.rgb_flatten = lambda x: torch.flatten(x, 1)

    def forward(self, rgb, ela, res, fft):
        feature_maps = self.rgb_features(rgb)
        
        pooled_rgb_features = self.rgb_avgpool(feature_maps)
        flattened_rgb_features = self.rgb_flatten(pooled_rgb_features)

        feat_ela = self.model.ela_branch(ela)
        feat_res = self.model.res_branch(res)
        feat_fft = self.model.fft_cnn(fft).flatten(1)
        
        fused = torch.cat((flattened_rgb_features, feat_ela, feat_res, feat_fft), dim=1)
        output = self.model.classifier(fused)
        
        return output, feature_maps

# Load the base model and then wrap it for CAM
base_model = DeepRevealModel4Branch().to(DEVICE)
base_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
base_model.eval()

model = DeepRevealWithCAM(base_model)
model.eval()
print("--- DeepReveal Model and CAM Wrapper Loaded Successfully ---")

# --- 4. Main Prediction Function ---
def get_prediction(input_image: Image.Image):
    original_img_pil = input_image.convert('RGB')
    
    # Make a copy and resize for consistency
    img_to_draw_on = np.array(original_img_pil.resize((224, 224))) 
    
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
    
    # Generate multimodal inputs
    ela_img = generate_ela(original_img_pil)
    res_img = generate_residual(original_img_pil)
    fft_img = generate_fft(original_img_pil)
    
    # Prepare tensors for the model
    rgb_tensor = data_transform(original_img_pil).unsqueeze(0).to(DEVICE)
    ela_tensor = data_transform(ela_img).unsqueeze(0).to(DEVICE)
    res_tensor = data_transform(res_img).unsqueeze(0).to(DEVICE)
    fft_tensor = fft_transform(fft_img).unsqueeze(0).to(DEVICE)
    
    # Initial prediction without gradient computation for overall efficiency
    with torch.no_grad():
        output_initial, _ = model(rgb_tensor, ela_tensor, res_tensor, fft_tensor) # '_' to ignore feature_maps here
    
    # --- Actual Prediction and Confidence ---
    probabilities = F.softmax(output_initial, dim=1)[0]
    idx_to_class = {0: 'fake', 1: 'real'}
    prediction_idx = probabilities.argmax().item()
    prediction_label = idx_to_class[prediction_idx].upper()
    confidences = {idx_to_class[i].upper(): f"{prob.item():.4f}" for i, prob in enumerate(probabilities)}

    # --- Grad-CAM Visualization Logic (ONLY if prediction is FAKE) ---
    if prediction_label == 'FAKE':
        try:
            # Re-run forward pass to get feature_maps WITH graph history for gradient computation
            # Create new tensors for this pass to avoid issues with `with torch.no_grad():`
            rgb_tensor_cam = data_transform(original_img_pil).unsqueeze(0).to(DEVICE)
            ela_tensor_cam = data_transform(ela_img).unsqueeze(0).to(DEVICE)
            res_tensor_cam = data_transform(res_img).unsqueeze(0).to(DEVICE)
            fft_tensor_cam = fft_transform(fft_img).unsqueeze(0).to(DEVICE)
            
            # Ensure model is in eval mode (already set globally, but good to double check)
            model.eval() 
            
            output_cam, feature_maps = model(rgb_tensor_cam, ela_tensor_cam, res_tensor_cam, fft_tensor_cam)
            
            # Get the score for the 'fake' class (index 0)
            class_to_idx = {'fake': 0, 'real': 1}
            target_class_score = output_cam[:, class_to_idx['fake']]
            
            # Zero all gradients before backward pass
            model.zero_grad() 
            
            # Compute gradients of the target score with respect to the feature maps
            grads = torch.autograd.grad(outputs=target_class_score, inputs=feature_maps, retain_graph=True)[0]
            
            # Pool the gradients across spatial dimensions (Height and Width)
            pooled_grads = torch.mean(grads, dim=[2, 3], keepdim=True)
            
            # Weight the feature maps with the pooled gradients and sum across channels
            heatmap = torch.sum(feature_maps * pooled_grads, dim=1).squeeze()
            
            # Apply ReLU to the heatmap (only positive contributions)
            # CORRECTED: Add .detach() before .cpu().numpy()
            heatmap_np = np.maximum(heatmap.detach().cpu().numpy(), 0)
            
            # Normalize the heatmap to 0-1
            max_val = np.max(heatmap_np)
            if max_val == 0:
                heatmap_np = np.zeros_like(heatmap_np) # If all zeros, keep it black
            else:
                heatmap_np = heatmap_np / max_val
            
            # Resize heatmap to 224x224 and apply JET colormap
            heatmap_resized = cv2.resize(heatmap_np, (224, 224))
            heatmap_colored = np.uint8(255 * heatmap_resized)
            heatmap_colored = cv2.applyColorMap(heatmap_colored, cv2.COLORMAP_JET)
            
            # Convert original image to BGR for OpenCV operations before overlay
            img_to_draw_on_bgr = cv2.cvtColor(img_to_draw_on, cv2.COLOR_RGB2BGR)
            
            # Overlay heatmap on the original image (50% transparency)
            overlay = cv2.addWeighted(img_to_draw_on_bgr, 0.5, heatmap_colored, 0.5, 0)
            
            # Convert the overlayed image back to RGB for PIL processing
            img_to_draw_on = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        except Exception as e:
            print(f"ERROR: Could not generate Grad-CAM heatmap due to an error: {e}")
            # If CAM fails, the original image (with prediction text) will be returned
            # `img_to_draw_on` retains its initial state from before the CAM attempt.
    
    # --- Draw Prediction Text on the Image ---
    # Ensure img_to_draw_on is BGR for cv2.putText before drawing
    img_for_text = cv2.cvtColor(img_to_draw_on, cv2.COLOR_RGB2BGR)
    color = (0, 0, 255) if "FAKE" in prediction_label else (0, 255, 0) # Red for FAKE, Green for REAL
    conf_text = f"Prediction: {prediction_label}"
    cv2.putText(img_for_text, conf_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Convert final image back to RGB for PIL and then Base64 encoding
    result_pil = Image.fromarray(cv2.cvtColor(img_for_text, cv2.COLOR_BGR2RGB))
    
    # --- Encode Result Image to Base64 ---
    buff = io.BytesIO()
    result_pil.save(buff, format="PNG")
    img_str = base64.b64encode(buff.getvalue()).decode("utf-8")
    
    return prediction_label, confidences, f"data:image/png;base64,{img_str}"

# --- 5. API Endpoints ---
class PredictionResponse(BaseModel):
    prediction: str
    confidence: dict
    result_image: str # Base64 encoded image string

@app.get("/")
def read_root():
    return {"message": "Welcome to the DeepReveal API. Please authenticate to use the /predict/ endpoint."}

@app.post("/predict/", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    # print(f"Authenticated user: {current_user.get('email', 'N/A')}") # Uncomment for auth debugging
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")
    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents))
        prediction, confidence, result_image_str = get_prediction(pil_image)
        return {"prediction": prediction, "confidence": confidence, "result_image": result_image_str}
    except Exception as e:
        print(f"An error occurred during prediction: {str(e)}") # Log the actual error
        raise HTTPException(status_code=500, detail=f"An error occurred during processing: {str(e)}")