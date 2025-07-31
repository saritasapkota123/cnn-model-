# File: crack_detection_cnn/predict_single.py

import torch
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont
import argparse
import os
import matplotlib.pyplot as plt
from models.model import HybridCNNTransformer
from termcolor import colored  # pip install termcolor

# -------------------- Argument Parser --------------------
parser = argparse.ArgumentParser(description="Hairline Crack Detection with Hybrid CNN + Transformer")
parser.add_argument("img_path", type=str, help="Path to the input image")
parser.add_argument("--save_result", action="store_true", help="Save annotated image with prediction")
parser.add_argument("--show", action="store_true", help="Display the image with label")
parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for crack classification")
parser.add_argument("--log", type=str, help="Path to log file for saving predictions")
args = parser.parse_args()

# -------------------- Load Model --------------------
model = HybridCNNTransformer()
model.load_state_dict(torch.load("models/hybrid_model.pth", map_location=torch.device("cpu")))
model.eval()

# -------------------- Image Preprocessing --------------------
image = Image.open(args.img_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
input_tensor = transform(image).unsqueeze(0)

# -------------------- Predict --------------------
with torch.no_grad():
    output = model(input_tensor)
    prob = torch.sigmoid(output).item()

is_crack = prob > args.threshold
label = "Crack Detected" if is_crack else "No Crack"
color = "red" if is_crack else "green"

# -------------------- Print Result --------------------
print(f"Prediction Probability: {prob:.4f}")
print(colored(f"Result: {label}", color.upper()))

# -------------------- Optional Logging --------------------
if args.log:
    with open(args.log, 'a') as f:
        f.write(f"{os.path.basename(args.img_path)}: {label} ({prob:.4f})\n")

# -------------------- Save Annotated Image --------------------
if args.save_result:
    original = Image.open(args.img_path).convert("RGB")
    draw = ImageDraw.Draw(original)

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except:
        font = ImageFont.load_default()

    draw.text((10, 10), f"{label} ({prob:.2f})", fill=color, font=font)
    result_path = f"result_{os.path.basename(args.img_path)}"
    original.save(result_path)
    print(f"🖼️ Annotated image saved as: {result_path}")

# -------------------- Optional Display --------------------
if args.show:
    plt.imshow(image)
    plt.title(f"{label} ({prob:.2f})")
    plt.axis('off')
    plt.show()
