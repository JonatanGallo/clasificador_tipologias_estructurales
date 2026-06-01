"""
train.py — Entrenamiento de clasificador de tipologías estructurales.

Backbones soportados (todos con pesos ImageNet vía torchvision o timm):
    densenet121       — baseline de Gómez et al. 2025
    resnet50          — estándar de la literatura
    efficientnetv2_s  — SOTA en eficiencia
    mobilenetv3_large — viabilidad de despliegue (MNRS)

Uso:
    python train.py \
        --data_root /ruta/datasets_finales/raw \
        --model densenet121 \
        --dataset_tag raw \
        --epochs 20 --batch 64 --lr 1e-4 \
        --out_dir runs/

Entradas esperadas (producidas por dataset_builder.py):
    data_root/{train,val,test}.csv  con columnas image_path,label,image_id,lat,lng,split
    data_root/{train,val,test}/<cls>/<id>.jpg

Salidas en out_dir/<run_name>/:
    best.pt           — checkpoint del mejor val_f1_macro
    last.pt           — último checkpoint
    history.csv       — métricas por época
    config.json       — configuración del run
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from PIL import Image
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


CLASSES = ["MAS", "PC", "DUAL"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ChapineroDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["label"].isin(CLASSES)].reset_index(drop=True)
        self.root_dir = Path(root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.root_dir / row["image_path"]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = CLASS_TO_IDX[row["label"]]
        return img, label


def get_transforms(img_size, train=True):
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(norm_mean, norm_std),
        ])


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
def build_model(name, num_classes, pretrained=True):
    name = name.lower()
    if name == "densenet121":
        from torchvision.models import densenet121, DenseNet121_Weights
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        m = densenet121(weights=weights)
        in_f = m.classifier.in_features
        m.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_f, num_classes)
        )
        return m

    if name == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = resnet50(weights=weights)
        in_f = m.fc.in_features
        m.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_f, num_classes)
        )
        return m

    if name == "efficientnetv2_s":
        from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights
        weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
        m = efficientnet_v2_s(weights=weights)
        in_f = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_f, num_classes)
        )
        return m

    if name == "mobilenetv3_large":
        from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        m = mobilenet_v3_large(weights=weights)
        in_f = m.classifier[3].in_features
        m.classifier[3] = nn.Linear(in_f, num_classes)
        return m

    if name == "vit_b_16":
        # IMPORTANTE: ViT espera 224x224. El loader debe usar img_size=224
        # cuando se entrena este modelo. train.py lo respeta vía --img_size 224.
        from torchvision.models import vit_b_16, ViT_B_16_Weights
        weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        m = vit_b_16(weights=weights)
        in_f = m.heads.head.in_features
        m.heads.head = nn.Linear(in_f, num_classes)
        return m

    raise ValueError(f"Modelo desconocido: {name}")


# ---------------------------------------------------------------------------
# Loops
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, device, train_mode):
    model.train(train_mode)
    losses, preds, labels = [], [], []
    ctx = torch.enable_grad() if train_mode else torch.no_grad()
    with ctx:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train_mode:
                optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device=="cuda")):
                logits = model(x)
                loss = criterion(logits, y)
            if train_mode:
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            preds.append(logits.argmax(dim=1).cpu().numpy())
            labels.append(y.cpu().numpy())

    preds = np.concatenate(preds) if preds else np.array([])
    labels = np.concatenate(labels) if labels else np.array([])
    if len(labels) == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(losses)),
        float(accuracy_score(labels, preds)),
        float(f1_score(labels, preds, average="macro", zero_division=0)),
    )


def class_weights_from_csv(csv_path):
    """Pesos inversos a la frecuencia de clase, normalizados al promedio."""
    df = pd.read_csv(csv_path)
    counts = df["label"].value_counts().to_dict()
    weights = []
    for c in CLASSES:
        n = counts.get(c, 1)
        weights.append(1.0 / n)
    w = np.array(weights, dtype=np.float32)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True, help="Raíz con train/val/test CSVs")
    p.add_argument("--model", required=True,
                   choices=["densenet121", "resnet50", "efficientnetv2_s",
                            "mobilenetv3_large", "vit_b_16"])
    p.add_argument("--dataset_tag", required=True, choices=["raw", "seg"])
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--img_size", type=int, default=300)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out_dir", default="runs")
    p.add_argument("--no_class_weights", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data_root = Path(args.data_root)
    run_name = f"{args.model}_{args.dataset_tag}_{int(time.time())}"
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Dataloaders
    train_ds = ChapineroDataset(data_root / "train.csv", data_root,
                                get_transforms(args.img_size, True))
    val_ds = ChapineroDataset(data_root / "val.csv", data_root,
                              get_transforms(args.img_size, False))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=args.num_workers, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=args.num_workers, pin_memory=(device == "cuda"))

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    # Modelo, pérdida, optimizador
    model = build_model(args.model, num_classes=len(CLASSES)).to(device)

    if args.no_class_weights:
        criterion = nn.CrossEntropyLoss()
    else:
        cw = class_weights_from_csv(data_root / "train.csv").to(device)
        print(f"Class weights ({CLASSES}): {cw.cpu().numpy().tolist()}")
        criterion = nn.CrossEntropyLoss(weight=cw)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",
                                                    factor=0.5, patience=2)

    history = []
    best_val_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, criterion,
                                           optimizer, device, train_mode=True)
        vl_loss, vl_acc, vl_f1 = run_epoch(model, val_loader, criterion,
                                           optimizer, device, train_mode=False)
        scheduler.step(vl_loss)
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc, "train_f1_macro": tr_f1,
            "val_loss": vl_loss, "val_acc": vl_acc, "val_f1_macro": vl_f1,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": round(elapsed, 1),
        })

        print(f"[{epoch:02d}/{args.epochs}] "
              f"train loss={tr_loss:.4f} acc={tr_acc:.3f} f1={tr_f1:.3f}  |  "
              f"val loss={vl_loss:.4f} acc={vl_acc:.3f} f1={vl_f1:.3f}  "
              f"({elapsed:.0f}s)")

        # Guarda último
        torch.save({"model_state": model.state_dict(),
                    "args": vars(args), "epoch": epoch},
                   run_dir / "last.pt")

        # Mejor por F1 macro de validación
        if vl_f1 > best_val_f1:
            best_val_f1 = vl_f1
            torch.save({"model_state": model.state_dict(),
                        "args": vars(args), "epoch": epoch,
                        "val_f1_macro": vl_f1},
                       run_dir / "best.pt")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping en época {epoch}")
                break

    # Guarda historial
    with open(run_dir / "history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    print(f"\nFin. Mejor val_f1_macro = {best_val_f1:.4f}  →  {run_dir}/best.pt")


if __name__ == "__main__":
    main()
