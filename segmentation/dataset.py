import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

VALID_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


class CrackDataset(Dataset):
    """
    Dataset for binary crack segmentation.

    Supports two layouts:
      1. Flat layout (DeepCrack):
           img_dir/  <- images
           mask_dir/ <- masks
         Pass img_dir and mask_dir explicitly.

      2. Subset layout (pothole-mix style):
           root_dir/subset_name/images/
           root_dir/subset_name/masks/
         Pass root_dir only; img_dir and mask_dir are ignored.
    """

    def __init__(self, img_dir=None, mask_dir=None, root_dir=None, transform=None):
        self.transform = transform
        self.image_paths = []
        self.mask_paths  = []

        if root_dir is not None:
            for subset in sorted(os.listdir(root_dir)):
                idir = os.path.join(root_dir, subset, 'images')
                mdir = os.path.join(root_dir, subset, 'masks')
                if not (os.path.isdir(idir) and os.path.isdir(mdir)):
                    continue
                imgs  = sorted(f for f in os.listdir(idir)
                               if os.path.splitext(f)[1].lower() in VALID_EXT)
                masks = sorted(f for f in os.listdir(mdir)
                               if os.path.splitext(f)[1].lower() in VALID_EXT)
                for img, msk in zip(imgs, masks):
                    self.image_paths.append(os.path.join(idir, img))
                    self.mask_paths.append(os.path.join(mdir, msk))
        else:
            imgs  = sorted(f for f in os.listdir(img_dir)
                           if os.path.splitext(f)[1].lower() in VALID_EXT)
            masks = sorted(f for f in os.listdir(mask_dir)
                           if os.path.splitext(f)[1].lower() in VALID_EXT)
            for img, msk in zip(imgs, masks):
                self.image_paths.append(os.path.join(img_dir, img))
                self.mask_paths.append(os.path.join(mask_dir, msk))

        print(f"[CrackDataset] Loaded {len(self)} samples")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert('RGB'))
        mask  = np.array(Image.open(self.mask_paths[idx]).convert('L'),
                         dtype=np.float32) / 255.0

        if self.transform:
            out   = self.transform(image=image, mask=mask)
            image = out['image']
            mask  = out['mask']

        return image, (mask > 0.5).long()


def get_transforms(image_size: int = 256):
    train_tf = A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # Affine replaces deprecated ShiftScaleRotate
        A.Affine(
            translate_percent={'x': (-0.0625, 0.0625), 'y': (-0.0625, 0.0625)},
            scale=(0.9, 1.1),
            rotate=(-45, 45),
            p=0.5,
        ),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, p=0.5),
        ], p=0.3),
        # CLAHE dramatically improves crack contrast — critical for thin crack visibility
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.RandomGamma(p=0.5),
        ], p=0.4),
        A.Sharpen(alpha=(0.1, 0.3), lightness=(0.9, 1.1), p=0.3),
        # CoarseDropout forces the model to use local context rather than memorising patches
        A.CoarseDropout(
            num_holes_range=(4, 8),
            hole_height_range=(16, 32),
            hole_width_range=(16, 32),
            fill=0,
            p=0.3,
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], is_check_shapes=False)

    val_tf = A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    return train_tf, val_tf


def build_dataloaders(
    train_img_dir: str,
    train_mask_dir: str,
    val_img_dir: str,
    val_mask_dir: str,
    batch_size: int = 8,
    num_workers: int = 2,
    image_size: int = 256,
):
    train_tf, val_tf = get_transforms(image_size)
    train_ds = CrackDataset(img_dir=train_img_dir, mask_dir=train_mask_dir, transform=train_tf)
    val_ds   = CrackDataset(img_dir=val_img_dir,   mask_dir=val_mask_dir,   transform=val_tf)

    # pin_memory only works on CUDA; disable on MPS/CPU to suppress the warning
    pin_memory = torch.cuda.is_available()

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0), drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )
    return train_dl, val_dl
