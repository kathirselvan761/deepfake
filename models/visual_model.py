import torch
import torch.nn as nn
import torchvision.models as models


class DeepfakeDetector(nn.Module):
    """
    EfficientNet-B0 based deepfake detector.
    
    Architecture:
    - Backbone: EfficientNet-B0 (pretrained on ImageNet)
    - Head: Linear(1280, 2) — Real vs Fake
    """
    
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        
        # Pre-trained EfficientNet-B0 load
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        
        # Last classifier layer replace
        # Original: Sequential(Dropout, Linear(1280, 1000))
        num_features = self.backbone.classifier[1].in_features  # 1280
        
        self.backbone.classifier[1] = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)


def get_model(device='cuda', num_classes=2):
    """Model create panni device ku move pannu."""
    model = DeepfakeDetector(num_classes=num_classes)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model Summary:")
    print(f"   Total params:     {total_params:,}")
    print(f"   Trainable params: {trainable_params:,}")
    print(f"   Device:           {device}")
    
    return model


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(device)
    
    # Test
    x = torch.randn(1, 3, 224, 224).to(device)
    out = model(x)
    
    print(f"\n✅ Test passed!")
    print(f"   Input:  {x.shape}")
    print(f"   Output: {out.shape}")  # Expected: (1, 2)