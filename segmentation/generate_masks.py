import os
import cv2
import numpy as np


import argparse


def resolve_paths(base_dir=None, image_dir=None, mask_dir=None, preview_dir=None):
    # Default relative dataset dir (relative to where the script is run)
    if base_dir is None:
        base_dir = "SematicSeg_Dataset"

    # If caller passed explicit image_dir use it; otherwise use base_dir + folder
    if image_dir is None:
        image_dir = os.path.join(base_dir, "Original Image")
    if mask_dir is None:
        # Use 'Labels' (capital L) which is used elsewhere in the repo
        mask_dir = os.path.join(base_dir, "Labels")
    if preview_dir is None:
        preview_dir = os.path.join(base_dir, "preview")

    # If paths don't exist relative to CWD, try resolving relative to repo root
    # (script is located in segmentation/). This permits running from segmentation/
    if not os.path.exists(image_dir):
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        alt_base = os.path.join(repo_root, base_dir)
        alt_image = os.path.join(alt_base, "Original Image")
        alt_mask = os.path.join(alt_base, "Labels")
        alt_preview = os.path.join(alt_base, "preview")
        if os.path.exists(alt_image):
            image_dir = alt_image
            mask_dir = alt_mask
            preview_dir = alt_preview

    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(preview_dir, exist_ok=True)
    return image_dir, mask_dir, preview_dir




def generate_crack_mask(image, min_area=150, open_iter=1, close_iter=1, canny_low=40, canny_high=120):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Increase contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Detect dark cracks
    blackhat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blackhat = cv2.morphologyEx(enhanced, cv2.MORPH_BLACKHAT, blackhat_kernel)

    # Threshold crack-like regions
    _, mask = cv2.threshold(
        blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Add edge information
    edges = cv2.Canny(enhanced, canny_low, canny_high)
    mask = cv2.bitwise_or(mask, edges)

    # Clean mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)

    # Remove tiny noise
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    clean_mask = np.zeros_like(mask)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean_mask[labels == i] = 255

    return clean_mask


def create_preview(image, mask):
    preview = image.copy()
    preview[mask == 255] = [0, 0, 255]  # red crack overlay
    return preview


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-dir', type=str, default=None)
    parser.add_argument('--image-dir', type=str, default=None)
    parser.add_argument('--mask-dir', type=str, default=None)
    parser.add_argument('--preview-dir', type=str, default=None)
    # Add CLI options for denoising and morphology
    parser.add_argument('--max-images', type=int, default=0,
                        help='Process only first N images (0 = all)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip images that already have mask files in mask-dir')
    parser.add_argument('--resize', type=int, default=0,
                        help='Resize images to this square size before processing (0 = keep original)')
    parser.add_argument('--min-area', type=int, default=150,
                        help='Minimum connected component area to keep (px)')
    parser.add_argument('--median', type=int, default=5,
                        help='Apply median blur with given kernel (0 = disable)')
    parser.add_argument('--bilateral', type=int, default=0,
                        help='Apply bilateral filter (diameter, 0 = disable)')
    parser.add_argument('--open-iter', type=int, default=1,
                        help='Morphological open iterations')
    parser.add_argument('--close-iter', type=int, default=1,
                        help='Morphological close iterations')
    parser.add_argument('--canny-low', type=int, default=40)
    parser.add_argument('--canny-high', type=int, default=120)
    args = parser.parse_args()

    IMAGE_DIR, MASK_DIR, PREVIEW_DIR = resolve_paths(args.base_dir, args.image_dir, args.mask_dir, args.preview_dir)

    print('Using image dir :', IMAGE_DIR)
    print('Using mask dir  :', MASK_DIR)
    print('Using preview dir:', PREVIEW_DIR)

    image_files = [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"))
    ]

    if len(image_files) == 0:
        print("ERROR: No images found in:")
        print(IMAGE_DIR)
        return

    print(f"Found {len(image_files)} images")
    # optionally limit images for quick testing
    if args.max_images and args.max_images > 0:
        image_files = image_files[: args.max_images]

    for idx, file_name in enumerate(image_files, 1):
        image_path = os.path.join(IMAGE_DIR, file_name)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        base_name = os.path.splitext(file_name)[0]
        mask_path = os.path.join(MASK_DIR, base_name + ".png")
        preview_path = os.path.join(PREVIEW_DIR, base_name + "_preview.png")

        # Skip if mask already exists (useful to resume)
        if args.skip_existing and os.path.exists(mask_path):
            if idx % 50 == 0:
                print(f"Skipping existing mask for {file_name}")
            continue

        # Optional resize for faster processing
        if args.resize and args.resize > 0:
            image = cv2.resize(image, (args.resize, args.resize), interpolation=cv2.INTER_AREA)

        # Optional pre-filtering
        if args.bilateral and args.bilateral > 0:
            image = cv2.bilateralFilter(image, args.bilateral, 75, 75)
        if args.median and args.median > 0:
            image = cv2.medianBlur(image, args.median)

        # Generate mask using CLI-configured thresholds and morphology
        mask = generate_crack_mask(image,
                                   min_area=args.min_area,
                                   open_iter=args.open_iter,
                                   close_iter=args.close_iter,
                                   canny_low=args.canny_low,
                                   canny_high=args.canny_high)

    # mask_path and preview_path already computed earlier

        try:
            written = cv2.imwrite(mask_path, mask)
            if not written:
                raise IOError('cv2.imwrite returned False')
        except Exception as e:
            print(f"Failed to write mask for {file_name}: {e}")
            continue

        preview = create_preview(image, mask)
        try:
            cv2.imwrite(preview_path, preview)
        except Exception:
            pass

        white_pixels = int(np.sum(mask == 255))

        if idx % 50 == 0 or idx == len(image_files):
            print(f"[{idx}/{len(image_files)}] Saved mask: {mask_path} | crack pixels: {white_pixels}")

    print("\nDone.")
    print("Masks saved in:", MASK_DIR)
    print("Preview images saved in:", PREVIEW_DIR)


if __name__ == "__main__":
    main()