import torch
import torch.nn as nn
from torchvision.models import resnet18
import timm

class HybridCNNTransformer(nn.Module):
    def __init__(self):
        super(HybridCNNTransformer, self).__init__()

        self.cnn = resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove the final classification layer (output: 512-dim)

        # ────────── Transformer Branch (Swin Transformer) ──────────
        self.transformer = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
        self.transformer.head = nn.Identity()  # Remove classifier head (output: 768-dim)

        # ────────── Final Classifier ──────────
        self.classifier = nn.Sequential(
            nn.Linear(512 + 768, 256),  
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)  
        )

    def forward(self, x):
        cnn_features = self.cnn(x)  # Output shape: (B, 512)
        transformer_features = self.transformer(x)  # Output shape: (B, 768)
        combined_features = torch.cat((cnn_features, transformer_features), dim=1)  # (B, 1280)
        return self.classifier(combined_features)  # Final output: (B, 1)

     
