import torch
from torchvision import transforms
from PIL import Image
import os
import sys

sys.path.append('.')
from models.visual_model import get_model

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = get_model(DEVICE)
state = torch.load('models/visual_best.pth', map_location=DEVICE)
model.load_state_dict(state)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

test_dir = 'data/raw/real_vs_fake/real-vs-fake/test'

def predict(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
    
    # ImageFolder: fake=0, real=1
    fake_prob = probs[0][0].item()
    real_prob = probs[0][1].item()
    
    print(f"{os.path.basename(image_path)}")
    print(f"  Fake: {fake_prob*100:.2f}% | Real: {real_prob*100:.2f}%")
    print(f"  → {'🔴 FAKE' if fake_prob > 0.5 else '🟢 REAL'}")
    print()

print("="*60)
print("REAL IMAGES (expect 🟢)")
print("="*60)
for f in os.listdir(f'{test_dir}/real')[:5]:
    predict(f'{test_dir}/real/{f}')

print("="*60)
print("FAKE IMAGES (expect 🔴)")
print("="*60)
for f in os.listdir(f'{test_dir}/fake')[:5]:
    predict(f'{test_dir}/fake/{f}')