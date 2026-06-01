"""
dataset_builder.py — Construye los datasets D-RAW y D-SEG con spatial split.

Toma las imágenes etiquetadas en /dataset/raw/<cls>/ y /dataset/seg/<cls>/
y produce CSVs train/val/test usando split espacial por grid de coordenadas
(las celdas completas se asignan a un único split, evitando que edificios
vecinos queden separados entre train y test).

Genera además, opcionalmente, el dataset D-SEG aplicando la máscara de fachada
sobre la imagen RAW si tu carpeta seg/ contiene máscaras binarias en lugar de
imágenes ya enmascaradas.

Uso:
    python dataset_builder.py \
        --dataset_dir /ruta/a/dataset \
        --labels_csv  /ruta/a/dataset/labels.csv \
        --out_dir     /ruta/a/datasets_finales \
        --grid_size   0.005 \
        --val_frac 0.15 --test_frac 0.15 \
        --img_size 300 --seed 42

Salida:
    out_dir/raw/{train,val,test}.csv
    out_dir/seg/{train,val,test}.csv
    out_dir/raw/{train,val,test}/<cls>/<id>.jpg
    out_dir/seg/{train,val,test}/<cls>/<id>.jpg
    out_dir/split_report.txt
"""

import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


# Clases que entran al clasificador (NC se excluye del entrenamiento)
TRAIN_CLASSES = ["MAS", "PC", "DUAL"]


def load_labels(labels_csv):
    rows = []
    with open(labels_csv, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                r["lat"] = float(r["lat"])
                r["lng"] = float(r["lng"])
            except (ValueError, KeyError):
                continue
            rows.append(r)
    return rows


def grid_split(rows, grid_size, val_frac, test_frac, seed):
    """Asigna cada celda del grid a un split. Todas las imágenes de la celda
    quedan en el mismo split."""
    rng = random.Random(seed)

    # Agrupa por celda
    cell_to_rows = defaultdict(list)
    for r in rows:
        cx = int(r["lat"] // grid_size)
        cy = int(r["lng"] // grid_size)
        cell_to_rows[(cx, cy)].append(r)

    cells = list(cell_to_rows.keys())
    rng.shuffle(cells)
    n = len(cells)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))

    test_cells = set(cells[:n_test])
    val_cells = set(cells[n_test:n_test + n_val])

    for r in rows:
        cell = (int(r["lat"] // grid_size), int(r["lng"] // grid_size))
        if cell in test_cells:
            r["split"] = "test"
        elif cell in val_cells:
            r["split"] = "val"
        else:
            r["split"] = "train"
    return rows


def apply_seg_mask(raw_path, seg_path, fill=128):
    """
    En este pipeline, Pipeline_chapinero.py escribe en seg/<cls>/ imágenes RGB
    ya enmascaradas (fachada aislada sobre fondo negro, 300x300). En ese caso,
    devolvemos la imagen tal cual.

    Como contingencia, si seg_path resulta ser una máscara binaria (pocos
    valores únicos en escala de grises), aplicamos la máscara sobre raw,
    redimensionándola al tamaño del raw con NEAREST para preservar el binario.
    """
    seg_pil = Image.open(seg_path)
    seg_gray = np.array(seg_pil.convert("L"))
    n_unique = len(np.unique(seg_gray))

    # Imagen RGB ya enmascarada (caso normal del pipeline actual)
    if n_unique > 10:
        return seg_pil.convert("RGB")

    # Máscara binaria (rama de contingencia)
    raw_arr = np.array(Image.open(raw_path).convert("RGB"))
    if seg_gray.shape != raw_arr.shape[:2]:
        seg_gray = np.array(
            seg_pil.convert("L").resize(
                (raw_arr.shape[1], raw_arr.shape[0]),
                Image.NEAREST,
            )
        )
    mask = (seg_gray > 127).astype(np.uint8)
    out = raw_arr * mask[:, :, None] + fill * (1 - mask[:, :, None])
    return Image.fromarray(out.astype(np.uint8))


def resize_and_save(src_pil_or_path, dst, size):
    if isinstance(src_pil_or_path, Image.Image):
        img = src_pil_or_path.convert("RGB")
    else:
        img = Image.open(src_pil_or_path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, quality=92)


def build_split_csv(rows, out_path):
    fields = ["image_path", "label", "image_id", "lat", "lng", "split"]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_dir", required=True,
                   help="Carpeta con raw/<cls> y seg/<cls> producida por etiquetador")
    p.add_argument("--labels_csv", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--grid_size", type=float, default=0.005,
                   help="Tamaño de celda en grados (~0.005 ≈ 500m en Bogotá)")
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.15)
    p.add_argument("--img_size", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exclude_low_conf", action="store_true",
                   help="Excluye imágenes marcadas como confianza baja")
    args = p.parse_args()

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)

    rows = load_labels(args.labels_csv)
    if args.exclude_low_conf:
        rows = [r for r in rows if r.get("low_confidence", "0") != "1"]

    # Filtra: solo MAS/PC/DUAL entran al entrenamiento (NC se descarta)
    rows = [r for r in rows if r["label"] in TRAIN_CLASSES]
    if not rows:
        raise SystemExit("No hay imágenes etiquetadas como MAS/PC/DUAL.")

    rows = grid_split(rows, args.grid_size, args.val_frac, args.test_frac, args.seed)

    # Reporte
    print("\n=== SPLIT REPORT ===")
    by_split = Counter(r["split"] for r in rows)
    print(f"Total entrenable: {len(rows)}  → {dict(by_split)}")
    by_split_class = defaultdict(Counter)
    for r in rows:
        by_split_class[r["split"]][r["label"]] += 1
    for split in ("train", "val", "test"):
        print(f"  {split:5s}  {dict(by_split_class[split])}")

    # Construye D-RAW y D-SEG
    for kind in ("raw", "seg"):
        kind_root = out_dir / kind
        kind_root.mkdir(parents=True, exist_ok=True)

        rows_with_paths = []
        for r in rows:
            cls = r["label"]
            iid = r["image_id"]

            # Localiza la imagen original
            src_dir = dataset_dir / kind / cls
            src = None
            for ext in (".jpg", ".png"):
                cand = src_dir / f"{iid}{ext}"
                if cand.exists():
                    src = cand
                    break
            if src is None:
                print(f"  ⚠ falta {kind}/{cls}/{iid} — se omite")
                continue

            # Para D-SEG, si la imagen no está enmascarada aún, aplica la máscara
            if kind == "seg":
                raw_src = dataset_dir / "raw" / cls / src.name
                if raw_src.exists():
                    pil = apply_seg_mask(raw_src, src)
                else:
                    pil = Image.open(src)
            else:
                pil = Image.open(src)

            dst = kind_root / r["split"] / cls / f"{iid}.jpg"
            resize_and_save(pil, dst, args.img_size)

            new_row = dict(r)
            new_row["image_path"] = str(dst.relative_to(out_dir))
            rows_with_paths.append(new_row)

        # CSVs por split
        for split in ("train", "val", "test"):
            split_rows = [r for r in rows_with_paths if r["split"] == split]
            build_split_csv(split_rows, kind_root / f"{split}.csv")
        print(f"  · {kind}: imágenes escritas en {kind_root}")

    # Reporte en disco
    with open(out_dir / "split_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Total entrenable: {len(rows)}\n")
        f.write(f"Por split: {dict(by_split)}\n")
        for split in ("train", "val", "test"):
            f.write(f"  {split}: {dict(by_split_class[split])}\n")
        f.write(f"Grid size: {args.grid_size}°  Seed: {args.seed}\n")
    print(f"\nListo. Salida en {out_dir}")


if __name__ == "__main__":
    main()
