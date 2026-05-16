#!/usr/bin/env python3
"""
Generate binary crack masks using a trained EnhancedGraphUNet checkpoint.

Output masks feed directly into Stage 2 (image-to-graph conversion).
White pixels = crack, black pixels = background.

Run:
    cd /Users/tejasskamar/Practicum/PHASE 2/crack-topology-gnn/segmentation
    python generate_masks.py \\
        --checkpoint-path ../outputs/segmentation/checkpoints/<best>.ckpt \\
        --input-dir  /path/to/images \\
        --output-dir /path/to/masks_output
"""

import os
import sys
import argparse

import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import EnhancedGraphUNet


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    model = EnhancedGraphUNet(in_channels=3, out_channels=2, features=(32, 64, 128, 256))
    checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint.get('state_dict', checkpoint)
    # Strip Lightning's 'model.' prefix if present
    clean = {(k[6:] if k.startswith('model.') else k): v for k, v in state_dict.items()}
    model.load_state_dict(clean)

    model.to(device)
    model.eval()
    return model


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device('mps')
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def parse_args():
    p = argparse.ArgumentParser(description='Generate crack masks with EnhancedGraphUNet')
    p.add_argument('--checkpoint-path', type=str, required=True)
    p.add_argument('--input-dir',  type=str,
                   default='/Users/tejasskamar/Practicum/Data Set/CrackDataset_DL_HY/split/val/images')
    p.add_argument('--output-dir', type=str,
                   default='/Users/tejasskamar/Practicum/PHASE 2/crack-topology-gnn/outputs/masks')
    p.add_argument('--image-size', type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    print(f'Using device: {device}')

    model = load_model(args.checkpoint_path, device)
    print('Model loaded.')

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    os.makedirs(args.output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(args.input_dir)
                   if os.path.splitext(f)[1].lower() in IMG_EXTS]
    print(f'Found {len(image_files)} images. Running inference...')

    with torch.no_grad():
        for fname in tqdm(image_files):
            img_path = os.path.join(args.input_dir, fname)
            try:
                original = Image.open(img_path).convert('RGB')
                orig_w, orig_h = original.size

                tensor = transform(original).unsqueeze(0).to(device)
                logits = model(tensor)                     # [1, 2, H, W]
                pred   = logits.argmax(dim=1).squeeze().cpu().numpy()  # [H, W]

                mask = (pred * 255).astype(np.uint8)
                mask = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

                out_name = os.path.splitext(fname)[0] + '.png'
                cv2.imwrite(os.path.join(args.output_dir, out_name), mask)

            except Exception as e:
                print(f'  Skipping {fname}: {e}')

    print(f'\nDone. Masks saved to: {args.output_dir}')
    print('These masks are ready for Stage 2: image-to-graph conversion.')


if __name__ == '__main__':
    main()
