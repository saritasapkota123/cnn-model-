# File: crack_detection_cnn/evaluate.py

import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from models.model import HybridCNNTransformer
import numpy as np

# Data and model loading
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
test_data = datasets.ImageFolder('data/processed/test', transform=transform)
test_loader = DataLoader(test_data, batch_size=16)

model = HybridCNNTransformer()
model.load_state_dict(torch.load("models/hybrid_model.pth", map_location=torch.device('cpu')))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        preds = torch.sigmoid(outputs).squeeze().numpy() > 0.5
        all_preds.extend(preds.astype(int))
        all_labels.extend(labels.numpy())

# Save classification report
report = classification_report(all_labels, all_preds, target_names=test_data.classes)
os.makedirs("results", exist_ok=True)
with open("results/evaluation_report.txt", "w") as f:
    f.write(report)

# Save confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_data.classes, yticklabels=test_data.classes)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.close()
