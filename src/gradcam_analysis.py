"""
gradcam_analysis.py — Interpretabilidad con Grad-CAM.

Genera mapas de activación que muestran QUÉ regiones de la fachada usa el
modelo para clasificar cada imagen. Es el core interpretativo de la tesis.

Soporta:
    - CNNs: DenseNet121, ResNet50, EfficientNetV2-S, MobileNetV3-Large
            → Grad-CAM clásico sobre la última capa convolucional
    - ViT-B/16
            → Grad-CAM adaptado para Transformers (reshape_transform)

Genera 4 outputs por imagen:
    1. Heatmap solo
    2. Overlay heatmap + imagen original
    3. Imagen original
    4. Imagen con la región top-30% más activa enmascarada (resto en gris)

Y un análisis agregado:
    - Heatmap promedio por clase (qué mira el modelo "en general" para MAS, PC, DUAL)
    - Análisis de errores: heatmaps de imágenes mal clasificadas
    - Frecuencia de activación por zona (fachada / cielo / suelo / vegetación)
      cruzando con la máscara SegFormer

Uso:
    python gradcam_analysis.py \
        --run_dir runs/densenet121_raw_1730000000 \
        --data_root datasets_finales/raw \
        --split test \
        --out_dir runs/densenet121_raw_1730000000/gradcam \
        --max_images 30 \
        --analyze_errors

Requisitos:
    pip install grad-cam   (se importa como pytorch_grad_cam)
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    raise SystemExit(
        "Falta el paquete grad-cam. Instálalo con:\n"
        "    pip install grad-cam"
    )

from train import build_model, CLASSES


CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}


# ---------------------------------------------------------------------------
# Configuración por arquitectura: capa target y reshape (para ViT)
# ---------------------------------------------------------------------------
def vit_reshape_transform(tensor, height=14, width=14):
    """Para ViT-B/16 con 224x224: 14x14 patches + 1 CLS token.
    Reordena los tokens (sin CLS) en una grilla espacial."""
    # tensor: [B, 197, D]  (1 CLS + 14*14 patches)
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
    # [B, H, W, D] → [B, D, H, W]
    return result.transpose(2, 3).transpose(1, 2)


def get_target_layers(model, model_name):
    """Devuelve la(s) capa(s) target y la función reshape (None para CNNs)."""
    name = model_name.lower()
    if name == "densenet121":
        return [model.features.denseblock4.denselayer16], None
    if name == "resnet50":
        return [model.layer4[-1]], None
    if name == "efficientnetv2_s":
        return [model.features[-1]], None
    if name == "mobilenetv3_large":
        return [model.features[-1]], None
    if name == "vit_b_16":
        # Última capa de la encoder antes del head
        return [model.encoder.layers[-1].ln_1], vit_reshape_transform
    raise ValueError(f"Modelo no soportado para Grad-CAM: {model_name}")


# ---------------------------------------------------------------------------
# Loaders y helpers
# ---------------------------------------------------------------------------
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


def get_eval_transform(img_size):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def denormalize_for_plot(tensor):
    """De tensor normalizado a numpy [H,W,3] en [0,1] para visualización."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = tensor.cpu().numpy().transpose(1, 2, 0)
    arr = arr * std + mean
    return np.clip(arr, 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Generación de Grad-CAM por imagen
# ---------------------------------------------------------------------------
def compute_gradcam(cam_obj, input_tensor, target_class):
    """Devuelve heatmap normalizado [H, W] en [0, 1]."""
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    targets = [ClassifierOutputTarget(target_class)]
    grayscale_cam = cam_obj(input_tensor=input_tensor, targets=targets)
    return grayscale_cam[0, :]  # [H, W]


def make_panel(img_rgb, heatmap, true_cls, pred_cls, prob, out_path, mask_top_pct=0.3):
    """Panel 2x2: original | heatmap | overlay | top-30% mask."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title(f"Original — real: {true_cls}")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(heatmap, cmap="jet")
    axes[0, 1].set_title("Grad-CAM heatmap")
    axes[0, 1].axis("off")

    overlay = show_cam_on_image(img_rgb, heatmap, use_rgb=True)
    axes[1, 0].imshow(overlay)
    correct = "✓" if true_cls == pred_cls else "✗"
    axes[1, 0].set_title(f"Overlay  ·  pred: {pred_cls} ({prob:.2f}) {correct}")
    axes[1, 0].axis("off")

    # Top-N% más activo
    thresh = np.quantile(heatmap, 1 - mask_top_pct)
    mask = (heatmap >= thresh).astype(np.float32)[..., None]
    masked = img_rgb * mask + 0.5 * (1 - mask)
    axes[1, 1].imshow(masked)
    axes[1, 1].set_title(f"Región top-{int(mask_top_pct*100)}% (lo que el modelo MIRA)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def make_class_average_panel(class_avg_heatmaps, class_counts, out_path):
    """Heatmap promedio por clase — qué mira el modelo "en general"."""
    classes_present = [c for c in CLASSES if class_counts.get(c, 0) > 0]
    n = len(classes_present)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]
    for ax, cls in zip(axes, classes_present):
        avg = class_avg_heatmaps[cls]
        ax.imshow(avg, cmap="jet")
        ax.set_title(f"{cls}  (n={class_counts[cls]})\nactivación promedio")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Análisis de zonas (cruce con máscara SegFormer si está disponible)
# ---------------------------------------------------------------------------
def analyze_zones(heatmap, seg_mask_path):
    if not seg_mask_path or not Path(seg_mask_path).exists():
        return None
    img = np.array(Image.open(seg_mask_path).convert("RGB").resize(
        (heatmap.shape[1], heatmap.shape[0]), Image.NEAREST)).astype(int)

    # Detecta el color de relleno tomando muestras de las 4 esquinas
    # (siempre son fondo en imágenes seg bien formadas)
    corners = np.concatenate([
        img[:5, :5].reshape(-1, 3),
        img[:5, -5:].reshape(-1, 3),
        img[-5:, :5].reshape(-1, 3),
        img[-5:, -5:].reshape(-1, 3),
    ])
    fill_color = np.median(corners, axis=0)  # (R, G, B) del fondo

    # Fondo = pixeles dentro de ±5 del color de relleno en los 3 canales
    is_background = np.all(np.abs(img - fill_color) <= 5, axis=-1)
    facade_mask = (~is_background).astype(np.float32)

    # Sanity check: si más del 95% es "fachada", no hay relleno detectable
    # (caso D-RAW) — eso está bien, no hay nada que excluir.
    h_norm = heatmap / (heatmap.sum() + 1e-8)
    pct_facade = float((h_norm * facade_mask).sum() * 100)
    return {
        "pct_activation_on_facade": pct_facade,
        "pct_activation_off_facade": 100 - pct_facade,
        "fill_color_detected": fill_color.tolist(),
        "pct_image_is_background": float(is_background.mean() * 100),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--out_dir", default=None)
    p.add_argument("--max_images", type=int, default=30,
                   help="Máximo de imágenes individuales a analizar (por orden)")
    p.add_argument("--per_class", type=int, default=10,
                   help="Imágenes por clase para promedios y galería")
    p.add_argument("--analyze_errors", action="store_true",
                   help="Genera panel adicional solo con imágenes mal clasificadas")
    p.add_argument("--seg_dir", default=None,
                   help="Opcional: carpeta con máscaras de fachada para análisis de zonas")
    p.add_argument("--mask_top_pct", type=float, default=0.3)
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "individual").mkdir(exist_ok=True)
    if args.analyze_errors:
        (out_dir / "errors").mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_args = load_model(run_dir, device)
    img_size = model_args.get("img_size", 300)

    # ViT requiere 224
    if model_args["model"] == "vit_b_16" and img_size != 224:
        print(f"⚠ ViT entrenado con img_size={img_size}. Forzando a 224 para CAM.")
        img_size = 224

    print(f"Modelo: {model_args['model']}  ·  Dataset: {model_args['dataset_tag']}")
    print(f"Device: {device}")

    target_layers, reshape_fn = get_target_layers(model, model_args["model"])
    cam_obj = GradCAM(model=model, target_layers=target_layers,
                      reshape_transform=reshape_fn)

    transform = get_eval_transform(img_size)

    # Carga split
    df = pd.read_csv(Path(args.data_root) / f"{args.split}.csv")
    df = df[df["label"].isin(CLASSES)].reset_index(drop=True)

    # Selecciona per_class imágenes por clase (sin groupby para evitar problemas con pandas 2.x)
    selected_parts = []
    for cls in CLASSES:
        sub = df[df["label"] == cls].head(args.per_class)
        selected_parts.append(sub)
    selected = pd.concat(selected_parts, ignore_index=True)
    if args.max_images:
        selected = selected.head(args.max_images)
        
    selected_parts = []
    for cls in CLASSES:
        sub = df[df["label"] == cls].head(args.per_class)
        selected_parts.append(sub)
    selected = pd.concat(selected_parts, ignore_index=True)
    if args.max_images:
        selected = selected.head(args.max_images)
    
    # Acumuladores para promedios y reporte
    class_heatmap_sum = {c: np.zeros((img_size, img_size), dtype=np.float64) for c in CLASSES}
    class_counts = {c: 0 for c in CLASSES}
    zone_records = []
    rows = []
    error_count = 0

    for i, row in selected.iterrows():
        img_path = Path(args.data_root) / row["image_path"]
        if not img_path.exists():
            continue
        true_cls = row["label"]
        true_idx = CLASS_TO_IDX[true_cls]

        pil = Image.open(img_path).convert("RGB").resize((img_size, img_size))
        img_rgb = np.array(pil).astype(np.float32) / 255.0

        x = transform(pil).unsqueeze(0).to(device)

        # Forward para predicción
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(probs.argmax())
        pred_cls = IDX_TO_CLASS[pred_idx]
        pred_prob = float(probs[pred_idx])
        is_error = (pred_cls != true_cls)

        # Grad-CAM con respecto a la clase REAL (no la predicha) — más informativo
        heatmap = compute_gradcam(cam_obj, x, target_class=true_idx)

        # Acumula para promedio por clase
        class_heatmap_sum[true_cls] += heatmap
        class_counts[true_cls] += 1

        # Análisis de zonas si hay máscara
        seg_mask_path = None
        if args.seg_dir:
            seg_mask_path = Path(args.seg_dir) / f"{row['image_id']}.jpg"
            if not seg_mask_path.exists():
                seg_mask_path = Path(args.seg_dir) / f"{row['image_id']}.png"
        zone = analyze_zones(heatmap, seg_mask_path) if args.seg_dir else None
        if zone:
            zone["image_id"] = row["image_id"]
            zone["label"] = true_cls
            zone["pred"] = pred_cls
            zone_records.append(zone)

        # Panel individual
        out_name = f"{i:03d}_{row['image_id']}_{true_cls}_to_{pred_cls}.png"
        target_dir = out_dir / "errors" if (is_error and args.analyze_errors) else out_dir / "individual"
        make_panel(img_rgb, heatmap, true_cls, pred_cls, pred_prob,
                   target_dir / out_name, mask_top_pct=args.mask_top_pct)
        if is_error:
            error_count += 1

        rows.append({
            "image_id": row["image_id"],
            "label": true_cls,
            "pred": pred_cls,
            "prob_pred": pred_prob,
            "correct": int(not is_error),
            "panel_path": str((target_dir / out_name).relative_to(out_dir)),
        })

    # Heatmaps promedio por clase
    class_avg = {c: (class_heatmap_sum[c] / class_counts[c]) if class_counts[c] > 0
                 else None for c in CLASSES}
    make_class_average_panel(class_avg, class_counts, out_dir / "class_average_heatmaps.png")

    # CSV con índice de paneles
    pd.DataFrame(rows).to_csv(out_dir / "gradcam_index.csv", index=False)

    # Reporte de zonas si aplica
    summary = {
        "model": model_args["model"],
        "dataset_tag": model_args["dataset_tag"],
        "split": args.split,
        "n_analyzed": len(rows),
        "n_errors": error_count,
        "class_counts": class_counts,
    }
    if zone_records:
        zdf = pd.DataFrame(zone_records)
        zdf.to_csv(out_dir / "zone_analysis.csv", index=False)
        summary["mean_pct_activation_on_facade"] = float(zdf["pct_activation_on_facade"].mean())
        # Por clase
        summary["pct_facade_by_class"] = (
            zdf.groupby("label")["pct_activation_on_facade"].mean().to_dict()
        )

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Análisis Grad-CAM completo")
    print(f"  · Imágenes analizadas: {len(rows)}  (errores: {error_count})")
    print(f"  · Por clase: {class_counts}")
    if zone_records:
        print(f"  · Activación promedio sobre fachada: "
              f"{summary['mean_pct_activation_on_facade']:.1f}%")
    print(f"  · Salidas en: {out_dir}/")


if __name__ == "__main__":
    main()
