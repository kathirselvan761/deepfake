import numpy as np
import cv2
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms


def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])


def generate_heatmap(model, image_path, device='cuda'):
    """Grad-CAM heatmap generate pannu."""
    model.eval()
    transform = get_transform()
    
    img = Image.open(image_path).convert('RGB')
    img_resized = img.resize((224, 224))
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Target layer — last conv layer
    target_layers = [model.backbone.features[-1]]
    
    # Prediction first — ethu class ah explain pannanum nu decide panna
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        # Class order: ['fake', 'real'] — ImageFolder alphabetical
        pred_class = probs.argmax(dim=1).item()
        fake_prob = probs[0][0].item()
    
    # Target: predicted class
    targets = [ClassifierOutputTarget(pred_class)]
    
    # Grad-CAM with targets
    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=img_tensor, targets=targets)[0]
    
    rgb_img = np.array(img_resized) / 255.0
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    return visualization, fake_prob


def save_heatmap(heatmap, output_path):
    """Heatmap save pannu."""
    bgr = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, bgr)
    print(f"💾 Saved: {output_path}")


if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from models.visual_model import get_model
    import os
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model(device)
    model.load_state_dict(torch.load('models/visual_best.pth', map_location=device))
    
    os.makedirs('outputs', exist_ok=True)
    
    test_dir = 'data/raw/real_vs_fake/real-vs-fake/test'
    
    # Test fake image
    fake_img = f'{test_dir}/fake/' + os.listdir(f'{test_dir}/fake')[0]
    heatmap, prob = generate_heatmap(model, fake_img, device)
    print(f"🔴 Fake image — Fake prob: {prob*100:.2f}%")
    save_heatmap(heatmap, 'outputs/heatmap_fake.jpg')
    
    # Test real image
    real_img = f'{test_dir}/real/' + os.listdir(f'{test_dir}/real')[0]
    heatmap, prob = generate_heatmap(model, real_img, device)
    print(f"🟢 Real image — Fake prob: {prob*100:.2f}%")
    save_heatmap(heatmap, 'outputs/heatmap_real.jpg')	