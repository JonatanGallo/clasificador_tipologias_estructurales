"""
etiquetador.py — Herramienta de etiquetado para tipologías estructurales en Chapinero.

Clases (esquema tipo Gómez et al. 2025, adaptado a Bogotá):
    MAS   = Mampostería (junta MNR + MC + mixto)
    PC    = Pórticos de Concreto (LFM en taxonomía GEM)
    DUAL  = Sistema dual (frame + muro de cortante)
    NC    = No clasificable (oclusión, jardín, garaje, fachada cubierta)

Uso:
    python etiquetador.py \
        --raw_dir /ruta/a/imagenes_crudas \
        --seg_dir /ruta/a/imagenes_segmentadas \
        --metadata metadata.csv \
        --out_dir /ruta/a/dataset

El CSV de metadata debe tener al menos: image_id, lat, lng
Opcional: year_construction, num_pisos, direccion

Teclas:
    1 = MAS    2 = PC    3 = DUAL    4 = NC
    n = siguiente sin etiquetar    b = anterior    s = skip (revisar después)
    c = marcar/desmarcar como confianza baja
    q = guardar y salir
"""

import argparse
import csv
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path
from tkinter import Tk, Label, Frame, StringVar, BOTH, LEFT, RIGHT, TOP, BOTTOM, X, Y
from tkinter import font as tkfont

try:
    from PIL import Image, ImageTk
except ImportError:
    print("ERROR: instala Pillow → pip install pillow")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuración de clases — única fuente de verdad del esquema
# ---------------------------------------------------------------------------
CLASSES = {
    "1": ("MAS", "Mampostería"),
    "2": ("PC", "Pórticos de Concreto"),
    "3": ("DUAL", "Sistema Dual"),
    "4": ("NC", "No Clasificable"),
}
CLASS_NAMES = [c[0] for c in CLASSES.values()]
CLASS_COLORS = {
    "MAS": "#e07a5f",
    "PC": "#81b29a",
    "DUAL": "#3d405b",
    "NC": "#9a9a9a",
}


class Etiquetador:
    def __init__(self, raw_dir, seg_dir, metadata_csv, out_dir, max_w=900, max_h=700):
        self.raw_dir = Path(raw_dir)
        self.seg_dir = Path(seg_dir)
        self.out_dir = Path(out_dir)
        self.max_w = max_w
        self.max_h = max_h

        # Crea las 4 carpetas de salida (raw y seg en paralelo)
        for cls in CLASS_NAMES:
            (self.out_dir / "raw" / cls).mkdir(parents=True, exist_ok=True)
            (self.out_dir / "seg" / cls).mkdir(parents=True, exist_ok=True)

        # CSV de etiquetas (apppend-friendly)
        self.labels_csv = self.out_dir / "labels.csv"
        self.metadata = self._load_metadata(metadata_csv)
        self.labeled = self._load_existing_labels()

        # Lista de imágenes pendientes
        self.queue = [m for m in self.metadata if m["image_id"] not in self.labeled]
        self.idx = 0
        if not self.queue:
            print("Todas las imágenes ya están etiquetadas. Nada que hacer.")
            sys.exit(0)
        print(f"Pendientes: {len(self.queue)} | Ya etiquetadas: {len(self.labeled)}")

        self._build_ui()

    # -----------------------------------------------------------------------
    # Carga de metadata y estado previo
    # -----------------------------------------------------------------------
    def _load_metadata(self, path):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        if not rows:
            print(f"ERROR: {path} está vacío.")
            sys.exit(1)
        if "image_id" not in rows[0]:
            print("ERROR: el CSV de metadata debe tener columna image_id.")
            sys.exit(1)
        return rows

    def _load_existing_labels(self):
        labeled = {}
        if self.labels_csv.exists():
            with open(self.labels_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    labeled[row["image_id"]] = row
        return labeled

    def _append_label(self, row):
        new_file = not self.labels_csv.exists()
        with open(self.labels_csv, "a", encoding="utf-8", newline="") as f:
            fields = ["image_id", "label", "lat", "lng", "year_construction",
                      "num_pisos", "low_confidence", "timestamp"]
            writer = csv.DictWriter(f, fieldnames=fields)
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    # -----------------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------------
    def _build_ui(self):
        self.root = Tk()
        self.root.title("Etiquetador NSR-10 / GEM — Chapinero")
        self.root.configure(bg="#1f1f23")

        title_font = tkfont.Font(family="Helvetica", size=14, weight="bold")
        meta_font = tkfont.Font(family="Helvetica", size=11)
        legend_font = tkfont.Font(family="Helvetica", size=12, weight="bold")

        # Cabecera con progreso
        header = Frame(self.root, bg="#1f1f23")
        header.pack(side=TOP, fill=X, padx=10, pady=(10, 4))
        self.progress_var = StringVar()
        Label(header, textvariable=self.progress_var, bg="#1f1f23", fg="white",
              font=title_font).pack(side=LEFT)

        self.confidence_var = StringVar(value="")
        Label(header, textvariable=self.confidence_var, bg="#1f1f23",
              fg="#f4a261", font=legend_font).pack(side=RIGHT)

        # Imágenes lado a lado
        images = Frame(self.root, bg="#1f1f23")
        images.pack(side=TOP, fill=BOTH, expand=True, padx=10)

        left = Frame(images, bg="#1f1f23")
        left.pack(side=LEFT, padx=8)
        Label(left, text="RAW (GSV)", bg="#1f1f23", fg="white",
              font=legend_font).pack()
        self.raw_label = Label(left, bg="#2a2a30")
        self.raw_label.pack()

        right = Frame(images, bg="#1f1f23")
        right.pack(side=LEFT, padx=8)
        Label(right, text="Fachada segmentada (SegFormer)", bg="#1f1f23",
              fg="white", font=legend_font).pack()
        self.seg_label = Label(right, bg="#2a2a30")
        self.seg_label.pack()

        # Metadata
        self.meta_var = StringVar()
        Label(self.root, textvariable=self.meta_var, bg="#1f1f23", fg="#cfcfd4",
              font=meta_font, justify=LEFT).pack(side=TOP, fill=X, padx=10, pady=4)

        # Leyenda de clases
        legend = Frame(self.root, bg="#1f1f23")
        legend.pack(side=TOP, fill=X, padx=10, pady=4)
        for key, (cls, name) in CLASSES.items():
            Label(
                legend, text=f"[{key}] {cls} — {name}",
                bg=CLASS_COLORS[cls], fg="white", font=legend_font,
                padx=10, pady=4
            ).pack(side=LEFT, padx=4)

        # Hint inferior
        Label(
            self.root,
            text="n=siguiente   b=anterior   s=skip   c=marca conf-baja   q=guardar y salir",
            bg="#1f1f23", fg="#9a9a9a", font=meta_font
        ).pack(side=BOTTOM, fill=X, padx=10, pady=6)

        # Bindings
        self.root.bind("<Key>", self._on_key)
        self._show_current()

    def _show_current(self):
        if self.idx >= len(self.queue):
            self.progress_var.set("¡Listo! Cierra la ventana o presiona q.")
            return

        item = self.queue[self.idx]
        image_id = item["image_id"]
        raw_path = self.raw_dir / f"{image_id}.jpg"
        if not raw_path.exists():
            raw_path = self.raw_dir / f"{image_id}.png"
        seg_path = self.seg_dir / f"{image_id}.jpg"
        if not seg_path.exists():
            seg_path = self.seg_dir / f"{image_id}.png"

        # Actualiza progreso
        self.progress_var.set(
            f"{self.idx + 1} / {len(self.queue)}   ·   id={image_id}"
        )

        # Muestra imágenes
        self._show_image(self.raw_label, raw_path)
        self._show_image(self.seg_label, seg_path)

        # Metadata
        meta_lines = []
        for key in ("lat", "lng", "year_construction", "num_pisos", "direccion"):
            if key in item and item[key] not in ("", None):
                meta_lines.append(f"{key} = {item[key]}")
        self.meta_var.set("   |   ".join(meta_lines) if meta_lines else "")

        # Reset confianza
        self.confidence_var.set("")
        self._low_conf = False

    def _show_image(self, widget, path):
        if not path.exists():
            widget.config(image="", text=f"FALTA\n{path.name}", fg="#f4a261")
            return
        img = Image.open(path)
        img.thumbnail((self.max_w // 2, self.max_h), Image.LANCZOS)
        tkimg = ImageTk.PhotoImage(img)
        widget.config(image=tkimg, text="")
        widget.image = tkimg  # evita garbage collection

    # -----------------------------------------------------------------------
    # Eventos
    # -----------------------------------------------------------------------
    def _on_key(self, event):
        k = event.keysym.lower()
        if k == "q":
            self.root.destroy()
            return
        if k == "n":
            self.idx = min(self.idx + 1, len(self.queue) - 1)
            self._show_current()
            return
        if k == "b":
            self.idx = max(self.idx - 1, 0)
            self._show_current()
            return
        if k == "s":
            # Mueve la imagen actual al final de la cola
            if self.idx < len(self.queue) - 1:
                self.queue.append(self.queue.pop(self.idx))
                self._show_current()
            return
        if k == "c":
            self._low_conf = not getattr(self, "_low_conf", False)
            self.confidence_var.set("⚠ confianza baja" if self._low_conf else "")
            return
        if k in CLASSES:
            cls, _ = CLASSES[k]
            self._save_label(cls)
            self.idx += 1
            self._show_current()

    # -----------------------------------------------------------------------
    # Guardado de etiqueta
    # -----------------------------------------------------------------------
    def _save_label(self, cls):
        if self.idx >= len(self.queue):
            return
        item = self.queue[self.idx]
        image_id = item["image_id"]

        # Copia ambas variantes a las carpetas correspondientes
        for kind, src_dir in (("raw", self.raw_dir), ("seg", self.seg_dir)):
            for ext in (".jpg", ".png"):
                src = src_dir / f"{image_id}{ext}"
                if src.exists():
                    dst = self.out_dir / kind / cls / f"{image_id}{ext}"
                    shutil.copy2(src, dst)
                    break

        # Append al CSV
        row = {
            "image_id": image_id,
            "label": cls,
            "lat": item.get("lat", ""),
            "lng": item.get("lng", ""),
            "year_construction": item.get("year_construction", ""),
            "num_pisos": item.get("num_pisos", ""),
            "low_confidence": "1" if getattr(self, "_low_conf", False) else "0",
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self._append_label(row)
        self.labeled[image_id] = row

    def run(self):
        self.root.mainloop()
        print(f"Sesión guardada. Total etiquetadas: {len(self.labeled)}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--raw_dir", required=True, help="Carpeta con imágenes GSV crudas")
    p.add_argument("--seg_dir", required=True, help="Carpeta con imágenes segmentadas (SegFormer)")
    p.add_argument("--metadata", required=True, help="CSV con image_id, lat, lng, año, pisos")
    p.add_argument("--out_dir", required=True, help="Carpeta destino con subcarpetas raw/<cls> y seg/<cls>")
    p.add_argument("--max_w", type=int, default=900)
    p.add_argument("--max_h", type=int, default=700)
    args = p.parse_args()

    app = Etiquetador(args.raw_dir, args.seg_dir, args.metadata,
                      args.out_dir, args.max_w, args.max_h)
    app.run()


if __name__ == "__main__":
    main()
