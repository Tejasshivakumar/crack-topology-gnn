from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as albu
from albumentations.pytorch import ToTensorV2


class CrackDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = Path(img_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        imgs = sorted([
            f.name for f in self.img_dir.glob("*")
            if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        ])
        mask_stems = {
            Path(m.name).stem: m.name
            for m in self.mask_dir.glob("*")
            if m.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
        }
        self.pairs = [
            (img, mask_stems[Path(img).stem])
            for img in imgs
            if Path(img).stem in mask_stems
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.pairs[idx]
        img = np.array(Image.open(self.img_dir / img_name).convert("RGB"))
        mask = np.array(Image.open(self.mask_dir / mask_name).convert("L"))
        mask = (mask > 127).astype("float32")

        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"]
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask)

        return img, mask


def get_train_transform():
    return albu.Compose([
        albu.Resize(512, 512),
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.3),
        albu.RandomRotate90(p=0.3),
        albu.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, p=0.4),
        albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transform():
    return albu.Compose([
        albu.Resize(512, 512),
        albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
