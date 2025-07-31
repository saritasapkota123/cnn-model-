import torch
from torchvision import transforms
from PIL import Image
import argparse
from models.model import HybridCNNTransformer

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("img_path", type=str, help="Path to the image")
args = parser.parse_args()

# Load model
model = HybridCNNTransformer()
model.load_state_dict(torch.load("models/hybrid_model.pth", map_location=torch.device('cpu')))
model.eval()

# Preprocess image
image = Image.open(args.img_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
input_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    prob = torch.sigmoid(output).item()

print(f"Prediction Probability: {prob:.4f}")
print("Crack Detected" if prob > 0.5 else "No Crack Detected")


print("hairline creack detected !")



# File: crack_detection_cnn/predict_single.py

import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import argparse
import os
from models.model import HybridCNNTransformer

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("img_path", type=str, help="Path to the image")
parser.add_argument("--save_result", action="store_true", help="Save image with prediction text")
args = parser.parse_args()

# Load model
model = HybridCNNTransformer()
model.load_state_dict(torch.load("models/hybrid_model.pth", map_location=torch.device('cpu')))
model.eval()

# Preprocess image
image = Image.open(args.img_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
input_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    prob = torch.sigmoid(output).item()

# Show result
print(f"Prediction Probability: {prob:.4f}")
if prob > 0.5:
    print("Hairline Crack Detected!")
    prediction_text = "Crack Detected"
else:
    print("No Crack Detected.")
    prediction_text = "No Crack"

# Optional: save image with prediction label
if args.save_result:
    # Re-open the original image (unresized)
    original_image = Image.open(args.img_path).convert("RGB")
    draw = ImageDraw.Draw(original_image)

    # Load default font
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    # Add text
    draw.text((10, 10), f"{prediction_text} ({prob:.2f})", fill="red" if prob > 0.5 else "green", font=font)

    # Save result image
    base_name = os.path.basename(args.img_path)
    save_path = f"result_{base_name}"
    original_image.save(save_path)
    print(f"🖼️ Result saved as: {save_path}")
