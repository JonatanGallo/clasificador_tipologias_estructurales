"""
evaluate.py — Evaluación del modelo entrenado sobre el set de test.

Produce:
    metrics.json              — accuracy, F1 macro, F1 por clase
    classification_report.txt — reporte sklearn completo
    confusion_matrix.png      — matriz de confusión (cuentas y porcentajes)
    predictions.csv           — predicciones por imagen (para análisis de errores)

Uso:
    python evaluate.py \
        --run_dir runs/densenet121_raw_1730000000 \
        --data_root /ruta/datasets_finales/raw \
        --split test
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from torch.utils.data import DataLoader

from train import ChapineroDataset, build_model, get_transforms, CLASSES


def load_model(run_dir, device):
    ckpt_path = Path(run_dir) / "best.pt"
    if not ckpt_path.exists():
        ckpt_path = Path(run_dir) / "last.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    model = build_model(args["model"], num_classes=len(CLASSES))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, args


def predict(model, loader, device):
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = probs.argmax(axis=1)
            all_preds.append(preds)
            all_labels.append(y.numpy())
            all_probs.append(probs)
    return (np.concatenate(all_preds),
            np.concatenate(all_labels),
            np.concatenate(all_probs))


def plot_confusion(cm, classes, out_path, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Conteos
    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"{title}\nConteos")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax)

    # Porcentajes por fila
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    ax = axes[1]
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("Porcentaje por fila (recall)")
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm_norm[i, j]*100:.0f}%", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, model_args = load_model(run_dir, device)
    img_size = model_args.get("img_size", 300)

    csv_path = Path(args.data_root) / f"{args.split}.csv"
    ds = ChapineroDataset(csv_path, args.data_root, get_transforms(img_size, train=False))
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False,
                        num_workers=args.num_workers)

    preds, labels, probs = predict(model, loader, device)

    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro", zero_division=0)
    f1_per_class = f1_score(labels, preds, average=None, zero_division=0,
                            labels=list(range(len(CLASSES))))
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASSES))))
    report_txt = classification_report(labels, preds, target_names=CLASSES,
                                       digits=3, zero_division=0)

    # Salidas
    metrics = {
        "model": model_args["model"],
        "dataset_tag": model_args["dataset_tag"],
        "split": args.split,
        "n_samples": int(len(labels)),
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_per_class": {c: float(f) for c, f in zip(CLASSES, f1_per_class)},
        "confusion_matrix": cm.tolist(),
        "classes": CLASSES,
    }

    with open(run_dir / f"metrics_{args.split}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(run_dir / f"classification_report_{args.split}.txt", "w") as f:
        f.write(report_txt)

    title = f"{model_args['model']} · {model_args['dataset_tag']} · {args.split}"
    plot_confusion(cm, CLASSES, run_dir / f"confusion_matrix_{args.split}.png", title)

    # Predicciones individuales
    df_test = pd.read_csv(csv_path)
    df_test = df_test[df_test["label"].isin(CLASSES)].reset_index(drop=True)
    df_test["pred"] = [CLASSES[i] for i in preds]
    for i, c in enumerate(CLASSES):
        df_test[f"prob_{c}"] = probs[:, i]
    df_test["correct"] = (df_test["label"] == df_test["pred"]).astype(int)
    df_test.to_csv(run_dir / f"predictions_{args.split}.csv", index=False)

    print(f"\n=== Resultados ({args.split}) ===")
    print(f"Modelo: {model_args['model']}  |  Dataset: {model_args['dataset_tag']}")
    print(f"Accuracy:  {acc:.4f}")
    print(f"F1 macro:  {f1_macro:.4f}")
    print(f"F1 por clase: {dict(zip(CLASSES, [round(float(x), 3) for x in f1_per_class]))}")
    print(report_txt)
    print(f"\nArtefactos en {run_dir}/")


if __name__ == "__main__":
    main()
