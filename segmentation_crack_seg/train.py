import os
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from models.hybrid_unet import GraphUNet
from crack_dataset import CrackDataset, get_train_transform, get_val_transform

# ─── Dataset ──────────────────────────────────────────────────────────────────
DATA_ROOT = Path("/Users/tejasskamar/Practicum/Data Set/crack_seg_clean/clean")
TRAIN_IMG  = DATA_ROOT / "train" / "images"
TRAIN_MASK = DATA_ROOT / "train" / "masks"
TEST_IMG   = DATA_ROOT / "test"  / "images"
TEST_MASK  = DATA_ROOT / "test"  / "masks"

# ─── Hyperparams ──────────────────────────────────────────────────────────────
EPOCHS      = 10
LR          = 1e-3
BATCH_SIZE  = 8
WARMUP      = 5
VAL_SPLIT   = 0.15   # fraction of training set used for validation
SEED        = 42


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        print("Using CUDA GPU")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Using Apple Silicon MPS")
        return torch.device("mps")
    print("Using CPU")
    return torch.device("cpu")


def compute_iou(pred, target, threshold=0.5):
    pred = (torch.sigmoid(pred) > threshold).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + 1e-6) / (union + 1e-6)


def compute_dice(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum()
    return (2 * inter + smooth) / (pred.sum() + target.sum() + smooth)


def combined_loss(pred, target, dice_w=0.5, bce_w=0.5):
    dice = 1 - compute_dice(pred, target)
    bce  = F.binary_cross_entropy_with_logits(pred, target)
    return dice_w * dice + bce_w * bce


def build_loaders():
    full_ds = CrackDataset(TRAIN_IMG, TRAIN_MASK, transform=None)
    n = len(full_ds)
    indices = list(range(n))
    random.shuffle(indices)
    val_n   = int(n * VAL_SPLIT)
    val_idx = indices[:val_n]
    tr_idx  = indices[val_n:]

    train_ds = CrackDataset(TRAIN_IMG, TRAIN_MASK, get_train_transform())
    val_ds   = CrackDataset(TRAIN_IMG, TRAIN_MASK, get_val_transform())

    train_loader = DataLoader(
        Subset(train_ds, tr_idx),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        pin_memory=False, drop_last=True
    )
    val_loader = DataLoader(
        Subset(val_ds, val_idx),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        pin_memory=False
    )
    print(f"Train: {len(tr_idx)} images  |  Val: {len(val_idx)} images")
    return train_loader, val_loader


def train(model, train_loader, val_loader):
    device = get_device()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS - WARMUP, eta_min=1e-6
    )

    history = {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": [], "epoch_time": []}
    best_iou = 0.0
    os.makedirs("checkpoints", exist_ok=True)

    print(f"\n{'='*60}")
    print(f"TRAINING {model.name}  |  epochs={EPOCHS}  lr={LR}  bs={BATCH_SIZE}")
    print(f"{'='*60}\n")

    for epoch in range(EPOCHS):
        t0 = time.time()

        # warmup
        if epoch < WARMUP:
            for g in optimizer.param_groups:
                g["lr"] = LR * (epoch + 1) / WARMUP

        # ── train ──────────────────────────────────────────────────────────
        model.train()
        tr_loss = tr_iou = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)
        for imgs, masks in pbar:
            imgs  = imgs.to(device)
            masks = masks.unsqueeze(1).to(device)
            optimizer.zero_grad()
            preds = model(imgs)
            loss  = combined_loss(preds, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
            tr_iou  += compute_iou(preds, masks).item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        tr_loss /= len(train_loader)
        tr_iou  /= len(train_loader)

        # ── validate ────────────────────────────────────────────────────────
        model.eval()
        v_loss = v_iou = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs  = imgs.to(device)
                masks = masks.unsqueeze(1).to(device)
                preds = model(imgs)
                v_loss += combined_loss(preds, masks).item()
                v_iou  += compute_iou(preds, masks).item()
        v_loss /= len(val_loader)
        v_iou  /= len(val_loader)

        if epoch >= WARMUP:
            scheduler.step()

        elapsed = time.time() - t0
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(v_loss)
        history["train_iou"].append(tr_iou)
        history["val_iou"].append(v_iou)
        history["epoch_time"].append(elapsed)

        print(f"Epoch {epoch+1:3d}/{EPOCHS}  "
              f"train loss={tr_loss:.4f} iou={tr_iou:.4f}  "
              f"val loss={v_loss:.4f} iou={v_iou:.4f}  "
              f"({elapsed/60:.1f} min)")

        if v_iou > best_iou:
            best_iou = v_iou
            torch.save(model.state_dict(), f"checkpoints/{model.name}_best.pth")
            print(f"  -> new best val IoU: {best_iou:.4f}")

    return history, best_iou


def test(model):
    device = get_device()
    model  = model.to(device).eval()
    ds     = CrackDataset(TEST_IMG, TEST_MASK, get_val_transform())
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    t_iou = t_dice = 0.0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Testing"):
            imgs  = imgs.to(device)
            masks = masks.unsqueeze(1).to(device)
            preds = model(imgs)
            t_iou  += compute_iou(preds, masks).item()
            t_dice += compute_dice(preds, masks).item()

    t_iou  /= len(loader)
    t_dice /= len(loader)
    print(f"\nTest IoU: {t_iou:.4f}   Test Dice: {t_dice:.4f}")
    return t_iou, t_dice


def save_plots(history, best_val_iou, test_iou, test_dice):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history["train_loss"], label="Train", alpha=0.7)
    axes[0, 0].plot(history["val_loss"],   label="Val",   linewidth=2)
    axes[0, 0].set_title("Loss"); axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend(); axes[0, 0].grid(True)

    axes[0, 1].plot(history["train_iou"], label="Train", alpha=0.7)
    axes[0, 1].plot(history["val_iou"],   label="Val",   linewidth=2)
    axes[0, 1].axhline(best_val_iou, color="r", linestyle="--",
                       label=f"Best: {best_val_iou:.4f}")
    axes[0, 1].set_title("IoU"); axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend(); axes[0, 1].grid(True)

    axes[1, 0].plot([t / 60 for t in history["epoch_time"]])
    axes[1, 0].set_title("Time/Epoch (min)"); axes[1, 0].grid(True)

    bars = ["Best Val IoU", "Test IoU", "Test Dice"]
    vals = [best_val_iou, test_iou, test_dice]
    axes[1, 1].bar(bars, vals, color=["steelblue", "orange", "green"])
    axes[1, 1].set_ylim([0, 1]); axes[1, 1].set_title("Metrics")
    axes[1, 1].grid(True, axis="y")

    plt.tight_layout()
    plt.savefig("training_results_graphunet_crackseg.png", dpi=150)
    print("Saved training_results_graphunet_crackseg.png")


def main():
    set_seed(SEED)

    for p in [TRAIN_IMG, TRAIN_MASK, TEST_IMG, TEST_MASK]:
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")

    train_loader, val_loader = build_loaders()

    model = GraphUNet()
    history, best_val_iou = train(model, train_loader, val_loader)

    print("\nLoading best checkpoint for test evaluation...")
    model.load_state_dict(torch.load(f"checkpoints/{model.name}_best.pth", map_location="cpu"))
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    test_iou, test_dice = test(model)
    save_plots(history, best_val_iou, test_iou, test_dice)

    results = {
        "Model":           "GraphUNet (Hybrid)",
        "Dataset":         "crack_seg_clean",
        "Best_Val_IoU":    best_val_iou,
        "Test_IoU":        test_iou,
        "Test_Dice":       test_dice,
        "Epochs":          len(history["train_loss"]),
        "Avg_min_per_epoch": (sum(history["epoch_time"]) / len(history["epoch_time"])) / 60,
    }
    df = pd.DataFrame([results])
    df.to_csv("results_graphunet_crackseg.csv", index=False)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(df.to_string(index=False))
    print("\nSaved: checkpoints/GraphUNet_best.pth")
    print("       training_results_graphunet_crackseg.png")
    print("       results_graphunet_crackseg.csv")


if __name__ == "__main__":
    main()
