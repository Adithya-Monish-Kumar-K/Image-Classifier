import os
from typing import Tuple
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataloader import default_collate
from torchvision import transforms, datasets
import os


class SafeImageFolder(datasets.ImageFolder):
    """ImageFolder that skips or marks corrupt images instead of raising an exception.

    If an image can't be opened, returns None so the custom collate function can drop it.
    """
    def __getitem__(self, index):  # type: ignore
        path, target = self.samples[index]
        try:
            sample = self.loader(path)
        except (UnidentifiedImageError, OSError, ValueError) as e:
            # Log once per corrupt file (can be extended to move/remove file)
            print(f"[WARN] Skipping corrupt image: {path} ({e})")
            return None
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        # Return a dummy zero-sized batch to avoid iterator StopIteration inside training loop;
        # caller should handle zero-sized batch (we'll skip automatically in train loop if needed)
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return default_collate(batch)


def build_transforms(img_size: int = 224):
    return {
        'train': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }


def create_dataloaders(
    data_dir: str,
    img_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 0,
    skip_corrupt: bool = True,
):
    """Create train/val dataloaders.

    skip_corrupt: if True uses SafeImageFolder + safe_collate to skip unreadable images.
    """
    transforms_dict = build_transforms(img_size)
    base_cls = SafeImageFolder if skip_corrupt else datasets.ImageFolder
    dataset = base_cls(root=data_dir, transform=transforms_dict['train'])

    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Set validation transform separately (random_split keeps original transform reference)
    val_dataset.dataset.transform = transforms_dict['val']

    collate_fn = safe_collate if skip_corrupt else None
    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn),
        'val': DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    }

    class_names = dataset.classes
    return dataloaders, class_names
