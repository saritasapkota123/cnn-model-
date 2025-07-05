# File: crack_detection_cnn/train.py

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import optim
from models.model import HybridCNNTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load data
train_data = datasets.ImageFolder('data/processed/train', transform=transform)
val_data = datasets.ImageFolder('data/processed/val', transform=transform)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)

# Device and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridCNNTransformer().to(device)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Evaluation function
def evaluate_model(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    return acc, prec, rec

# Training loop
for epoch in range(10):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.float().unsqueeze(1).to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    acc, prec, rec = evaluate_model(model, val_loader)
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}, Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}")

# Save model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/hybrid_model.pth")
