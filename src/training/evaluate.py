import torch
from sklearn.metrics import classification_report, confusion_matrix
from src.utils import load_checkpoint, get_device
from src.models.model import build_model
from src.data.dataset import build_transforms
from torchvision import datasets
from torch.utils.data import DataLoader


def load_model(model_path: str):
    device = get_device()
    checkpoint = load_checkpoint(model_path, map_location=device)
    class_names = checkpoint['classes']
    model = build_model(num_classes=len(class_names), freeze_backbone=False)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    return model, class_names, device


def evaluate_model(model_path: str, data_dir: str, img_size: int = 224, batch_size: int = 32):
    model, class_names, device = load_model(model_path)
    transforms_dict = build_transforms(img_size)
    dataset = datasets.ImageFolder(root=data_dir, transform=transforms_dict['val'])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(labels.tolist())

    print(classification_report(all_targets, all_preds, target_names=class_names))
    print("Confusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    evaluate_model(args.model_path, args.data_dir, args.img_size, args.batch_size)
