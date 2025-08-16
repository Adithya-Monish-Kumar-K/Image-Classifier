import argparse
import os
from time import time
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from src.data.dataset import create_dataloaders
from src.models.model import build_model, get_optimizer, get_scheduler
from src.utils import set_seed, get_device, ensure_dir, save_checkpoint, count_parameters


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    losses = []
    preds_all = []
    targets_all = []
    for images, labels in dataloader:
        # Skip empty batches (could happen if all items in a batch were corrupt and filtered out)
        if images.numel() == 0:
            continue
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds = outputs.argmax(dim=1)
        preds_all.extend(preds.detach().cpu().tolist())
        targets_all.extend(labels.detach().cpu().tolist())
    acc = accuracy_score(targets_all, preds_all)
    return sum(losses)/len(losses), acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    losses = []
    preds_all = []
    targets_all = []
    with torch.no_grad():
        for images, labels in dataloader:
            if images.numel() == 0:
                continue
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            preds = outputs.argmax(dim=1)
            preds_all.extend(preds.detach().cpu().tolist())
            targets_all.extend(labels.detach().cpu().tolist())
    acc = accuracy_score(targets_all, preds_all)
    return sum(losses)/len(losses), acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--unfreeze', action='store_true', help='If set, unfreeze backbone regardless of freeze_backbone flag')
    parser.add_argument('--output_dir', type=str, default='models')
    parser.add_argument('--no_skip_corrupt', action='store_true', help='Process all images and error on corrupt ones')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    dataloaders, class_names = create_dataloaders(
        args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        skip_corrupt=not args.no_skip_corrupt,
    )
    model = build_model(num_classes=len(class_names), freeze_backbone=args.freeze_backbone and not args.unfreeze)

    if args.unfreeze:
        for p in model.parameters():
            p.requires_grad = True

    model.to(device)
    print(f"Trainable parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, lr=args.lr)
    scheduler = get_scheduler(optimizer)

    best_acc = 0.0
    ensure_dir(args.output_dir)
    best_model_path = os.path.join(args.output_dir, 'best_model.pt')

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, dataloaders['train'], criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, dataloaders['val'], criterion, device)
        scheduler.step()
        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint({'model_state': model.state_dict(), 'classes': class_names}, best_model_path)
            print(f"New best model saved with val acc {best_acc:.4f}")

    print(f"Training complete. Best Val Acc: {best_acc:.4f}. Model path: {best_model_path}")


if __name__ == '__main__':
    main()
