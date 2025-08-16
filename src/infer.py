import argparse
import torch
from PIL import Image
from torchvision import transforms
from src.utils import load_checkpoint, get_device
from src.models.model import build_model


def load_model(model_path: str):
    device = get_device()
    checkpoint = load_checkpoint(model_path, map_location=device)
    class_names = checkpoint['classes']
    model = build_model(num_classes=len(class_names), freeze_backbone=False)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    return model, class_names, device


def predict_image(model, class_names, device, image_path: str, img_size: int = 224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229,0.224,0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        pred_class = class_names[pred_idx]
        confidence = probs[0, pred_idx].item()
    return pred_class, confidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=224)
    args = parser.parse_args()

    model, class_names, device = load_model(args.model_path)
    pred_class, confidence = predict_image(model, class_names, device, args.image, args.img_size)
    print(f"Prediction: {pred_class} (confidence: {confidence:.4f})")

if __name__ == '__main__':
    main()
