import os
import argparse
import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from PIL import Image
import time from facebook
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class HybridCNNTransformer(nn.Module):
    def __init__(self):
        super(HybridCNNTransformer, self).__init__()
        self.cnn = resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()

        self.transformer = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
        self.transformer.head = nn.Identity()

        self.classifier = nn.Sequential(
            nn.Linear(512 + 768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        cnn_features = self.cnn(x)
        transformer_features = self.transformer(x)
        features = torch.cat((cnn_features, transformer_features), dim=1)
        return self.classifier(features)
def save_confusion_matrix(y_true, y_pred, class_names, filename="results/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def save_classification_report(y_true, y_pred, class_names, filename="results/evaluation_report.txt"):
    report = classification_report(y_true, y_pred, target_names=class_names)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as f:
        f.write(report)

def evaluate_model(model, val_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    return acc, prec, rec

def train_model():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_data = datasets.ImageFolder('data/processed/train', transform=transform)
    val_data = datasets.ImageFolder('data/processed/val', transform=transform)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)

    model = HybridCNNTransformer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

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

        acc, prec, rec = evaluate_model(model, val_loader, device)
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}, Val Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}")

    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/hybrid_model.pth")
    print("Training complete and model saved.")

def evaluate():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_data = datasets.ImageFolder('data/processed/test', transform=transform)
    test_loader = DataLoader(test_data, batch_size=16)

    model = HybridCNNTransformer()
    model.load_state_dict(torch.load("models/hybrid_model.pth", map_location=torch.device('cpu')))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            preds = torch.sigmoid(outputs).squeeze().numpy() > 0.5
            all_preds.extend(preds.astype(int))
            all_labels.extend(labels.numpy())

    save_confusion_matrix(all_labels, all_preds, test_data.classes)
    save_classification_report(all_labels, all_preds, test_data.classes)
    print("Evaluation complete. Report and confusion matrix saved.")


def batch_infer():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    model = HybridCNNTransformer()
    model.load_state_dict(torch.load("models/hybrid_model.pth", map_location=torch.device('cpu')))
    model.eval()

    input_folder = "data/processed/infer"
    output_file = "results/inference_results.csv"
    os.makedirs("results", exist_ok=True)

    with open(output_file, "w") as f:
        f.write("Image,Probability,Label\n")
        for filename in os.listdir(input_folder):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                path = os.path.join(input_folder, filename)
                image = Image.open(path).convert("RGB")
                input_tensor = transform(image).unsqueeze(0)
                with torch.no_grad():
                    output = model(input_tensor)
                    prob = torch.sigmoid(output).item()
                label = "Crack" if prob > 0.5 else "No Crack"
                f.write(f"{filename},{prob:.4f},{label}\n")

    print("Batch inference complete. Results saved to:", output_file)

def predict_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)

    model = HybridCNNTransformer()
    model.load_state_dict(torch.load("models/hybrid_model.pth", map_location=torch.device('cpu')))
    model.eval()

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()
        print(f"🔍 Probability: {prob:.4f}")
        if prob > 0.5:
            print("⚠️ Crack Detected")
        else:
            print("No Crack Detected")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hairline Crack Detection")
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'infer', 'predict'], required=True, help="Mode to run")
    parser.add_argument('--image', type=str, help="Path to image for single prediction")

    args = parser.parse_args()

    if args.mode == 'train':
        train_model()
    elif args.mode == 'evaluate':
        evaluate()
    elif args.mode == 'infer':
        batch_infer()
    elif args.mode == 'predict' and args.image:
        predict_image(args.image)
    else:
        print("Invalid usage. Use --help to see options.")
