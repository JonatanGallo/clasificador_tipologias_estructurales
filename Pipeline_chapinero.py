#!/usr/bin/env python3
"""
pipeline_chapinero.py  v1.0
════════════════════════════════════════════════════════════════════
Pipeline integrado: Descarga GSV → SegFormer → Estructura de etiquetado

MODO TEST (por defecto):
  python pipeline_chapinero.py --mode test
  → Descarga 10 edificios, corre SegFormer, genera estructura de carpetas

MODO PRODUCCIÓN:
  python pipeline_chapinero.py --mode prod --target 500
  → Descarga 500 edificios (~400 imágenes útiles tras filtros)

SOLO SEGMENTAR (si ya descargaste):
  python pipeline_chapinero.py --mode segment_only --input_dir imagenes_descargadas

ESTRUCTURA DE SALIDA:
  data/piloto/
  ├── originales/          ← imágenes crudas descargadas
  ├── segmentadas/         ← fachada aislada (fondo negro) 300×300
  ├── tripticos/           ← visualización [original|segmentación|overlay]
  ├── MAS/                 ← Mampostería (etiquetado manual)
  │   ├── originales/
  │   └── segmentadas/
  ├── DUAL/                ← Sistema dual
  │   ├── originales/
  │   └── segmentadas/
  ├── PC/                  ← Pórticos de concreto
  │   ├── originales/
  │   └── segmentadas/
  ├── NC/                  ← No clasificable / descarte
  │   ├── originales/
  │   └── segmentadas/
  ├── download_report.csv  ← Reporte de descarga con metadatos
  └── segmentation_report.csv ← Reporte de segmentación

USO PARA ETIQUETAR:
  1. Revisa las imágenes en originales/ y segmentadas/
  2. MUEVE (no copies) cada imagen a la carpeta de su clase
  3. Mueve AMBAS versiones (original + segmentada) a la misma clase
  4. Las imágenes malas van a NC/

Basado en: imagenes_crudas.py v3.0 + segmentacion_unificada.py
Referencia: Gomez et al. 2025 (MAS / LFM / LDUAL / TW)
Clases adaptadas: MAS, DUAL, PC, NC
"""

import os
import re
import sys
import math
import time
import json
import hashlib
import argparse
import requests
import numpy as np
import pandas as pd
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv()

# ════════════════════════════════════════════════════════════════
# CONFIGURACIÓN GLOBAL
# ════════════════════════════════════════════════════════════════

class Config:
    # ── API ──────────────────────────────────────────────────
    API_KEY = os.environ.get("GSV_API_KEY", "") # ← REEMPLAZA con tu API key
    if not Config.API_KEY:
        raise RuntimeError("Falta GSV_API_KEY. Copia .env.example a .env y pon tu clave.")
    # ── Rutas ────────────────────────────────────────────────
    BASE_DIR    = Path(__file__).resolve().parent
    CSV_PATH    = BASE_DIR / "coordenadas_pendientes.csv"
    OUTPUT_BASE = BASE_DIR / "data" / "piloto"

    # Subdirectorios de salida
    DIR_ORIGINALES  = OUTPUT_BASE / "originales"
    DIR_SEGMENTADAS = OUTPUT_BASE / "segmentadas"
    DIR_TRIPTICOS   = OUTPUT_BASE / "tripticos"

    # Clases para etiquetado (carpetas)
    CLASES = ["MAS", "DUAL", "PC", "NC"]

    # ── Descarga GSV ─────────────────────────────────────────
    TARGET_COUNT    = 600       # Test: 10, Producción: 500+
    MAX_WORKERS     = 6
    N_ANGULOS       = 3         # Ángulos por edificio
    DELTA_ANGULO    = 20        # Grados de desviación lateral
    GUARDAR_TODOS   = False     # Solo guardar la mejor vista
    SIZE            = "640x640"
    FOV_BASE        = 90
    FOV_MIN         = 70
    FOV_MAX         = 110
    MAX_PANO_DIST_M = 55
    AÑO_MIN_PANO    = 2018
    MIN_FECHA_REC   = 2020

    # ── IDECA ────────────────────────────────────────────────
    IDECA_TIMEOUT = 8
    IDECA_URL = (
        "https://serviciosgis.catastrobogota.gov.co/arcgis/rest/services/"
        "catastro/construccion/FeatureServer/0/query"
    )

    # ── Tipos OSM excluidos ──────────────────────────────────
    EXCLUDE_TYPES = {
        "construction", "ruins", "demolished", "ruin", "no",
        "shed", "garage", "carport", "hut", "static_caravan",
        "container", "tent",
    }

    # ── SegFormer ────────────────────────────────────────────
    SEGFORMER_MODEL = "nvidia/segformer-b0-finetuned-cityscapes-512-1024"
    CNN_SIZE        = (300, 300)    # Gomez et al. 2025 usa 300×300
    UMBRAL_MIN_FACHADA = 0.08      # Mínimo 8% de fachada
    UMBRAL_MAX_FACHADA = 0.95      # Máximo 95%
    MIN_CIELO_CALLE    = 0.02      # Algo de contexto exterior

    # ── URLs GSV ─────────────────────────────────────────────
    STREETVIEW_URL = "https://maps.googleapis.com/maps/api/streetview"
    METADATA_URL   = "https://maps.googleapis.com/maps/api/streetview/metadata"

    @classmethod
    def setup_dirs(cls):
        """Crea toda la estructura de carpetas."""
        for d in [cls.DIR_ORIGINALES, cls.DIR_SEGMENTADAS, cls.DIR_TRIPTICOS]:
            d.mkdir(parents=True, exist_ok=True)

        for clase in cls.CLASES:
            (cls.OUTPUT_BASE / clase / "originales").mkdir(parents=True, exist_ok=True)
            (cls.OUTPUT_BASE / clase / "segmentadas").mkdir(parents=True, exist_ok=True)

        print(f"[OK] Estructura de carpetas creada en: {cls.OUTPUT_BASE}")


# ════════════════════════════════════════════════════════════════
# UTILIDADES GEOESPACIALES
# ════════════════════════════════════════════════════════════════

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6_371_000
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat/2)**2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon/2)**2)
    return 2 * R * math.asin(math.sqrt(a))


def calculate_heading(lat1, lon1, lat2, lon2):
    dlon = math.radians(lon2 - lon1)
    lat1r, lat2r = math.radians(lat1), math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2r)
    y = (math.cos(lat1r) * math.sin(lat2r)
         - math.sin(lat1r) * math.cos(lat2r) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def dynamic_pitch(dist_m):
    if dist_m < 8:   return 20
    if dist_m < 15:  return 15
    if dist_m < 25:  return 10
    if dist_m < 40:  return  5
    return 2


def dynamic_fov(dist_m, ancho_m=None):
    if ancho_m and ancho_m > 0:
        fov_ideal = 2 * math.degrees(math.atan((ancho_m * 0.65) / max(dist_m, 1)))
        return max(Config.FOV_MIN, min(Config.FOV_MAX, round(fov_ideal)))
    if dist_m < 10:  return Config.FOV_MIN
    if dist_m < 20:  return 80
    if dist_m < 35:  return Config.FOV_BASE
    return min(Config.FOV_MAX, int(Config.FOV_BASE + (dist_m - 35) * 0.5))


# ════════════════════════════════════════════════════════════════
# POLÍGONOS IDECA
# ════════════════════════════════════════════════════════════════

def consultar_poligono_ideca(lat, lon):
    """Consulta polígono del predio en IDECA. Retorna dict o None."""
    delta = 0.0002
    params = {
        "where":             "1=1",
        "geometry":          f"{lon-delta},{lat-delta},{lon+delta},{lat+delta}",
        "geometryType":      "esriGeometryEnvelope",
        "inSR":              "4326",
        "outSR":             "4326",
        "spatialRel":        "esriSpatialRelIntersects",
        "outFields":         "ANIO_CONSTR,NUM_PISOS,TIPO_CONST,ESTRATO",
        "returnGeometry":    "true",
        "f":                 "geojson",
        "resultRecordCount": 5,
    }
    try:
        r = requests.get(Config.IDECA_URL, params=params, timeout=Config.IDECA_TIMEOUT)
        if r.status_code != 200:
            return None
        feats = r.json().get("features", [])
        if not feats:
            return None

        mejor, mejor_dist = None, float("inf")
        for feat in feats:
            geom = feat.get("geometry", {})
            if geom.get("type") not in ("Polygon", "MultiPolygon"):
                continue
            coords = _extraer_coordenadas(geom)
            if not coords:
                continue
            cx = np.mean([c[0] for c in coords])
            cy = np.mean([c[1] for c in coords])
            d = haversine_m(lat, lon, cy, cx)
            if d < mejor_dist:
                mejor_dist = d
                mejor = (feat, coords, cx, cy)

        if mejor is None:
            return None

        feat, coords, cx, cy = mejor
        props = feat.get("properties", {})
        ancho_m, heading_fachada = _calcular_fachada_principal(coords, lat, lon)

        return {
            "ancho_m":          ancho_m,
            "heading_fachada":  heading_fachada,
            "ANIO_CONSTR":      props.get("ANIO_CONSTR"),
            "NUM_PISOS":        props.get("NUM_PISOS"),
            "TIPO_CONST":       props.get("TIPO_CONST"),
            "ESTRATO":          props.get("ESTRATO"),
            "dist_centroide_m": round(mejor_dist, 1),
        }
    except Exception:
        return None


def _extraer_coordenadas(geom):
    t = geom.get("type")
    if t == "Polygon":
        return geom["coordinates"][0]
    if t == "MultiPolygon":
        return geom["coordinates"][0][0]
    return []


def _calcular_fachada_principal(coords, pano_lat, pano_lon):
    if len(coords) < 3:
        return None, None
    mejor_seg, mejor_dist = None, float("inf")
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        cmid_lat = (lat1 + lat2) / 2
        cmid_lon = (lon1 + lon2) / 2
        d = haversine_m(pano_lat, pano_lon, cmid_lat, cmid_lon)
        if d < mejor_dist:
            mejor_dist = d
            mejor_seg = (lon1, lat1, lon2, lat2, cmid_lat, cmid_lon)
    if mejor_seg is None:
        return None, None
    lon1, lat1, lon2, lat2, cmid_lat, cmid_lon = mejor_seg
    ancho_m = haversine_m(lat1, lon1, lat2, lon2)
    heading = calculate_heading(pano_lat, pano_lon, cmid_lat, cmid_lon)
    return round(ancho_m, 1), round(heading, 1)


# ════════════════════════════════════════════════════════════════
# DESCARGA GSV
# ════════════════════════════════════════════════════════════════

def get_metadata(lat, lon):
    try:
        r = requests.get(Config.METADATA_URL,
                         params={"location": f"{lat},{lon}",
                                 "key": Config.API_KEY,
                                 "source": "outdoor"},
                         timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def validar_panorama(metadata, target_lat, target_lon):
    """Valida panorama ANTES de descargar (metadata es gratis)."""
    if not metadata or metadata.get("status") != "OK":
        return False, 0.0, "Sin cobertura", {}

    copyright_str = metadata.get("copyright", "")
    if "Google" not in copyright_str:
        return False, 0.0, f"Indoor/Business: '{copyright_str}'", {}

    pano_id = metadata.get("pano_id", "")
    if pano_id and not re.match(r'^[A-Za-z0-9_\-]{20,}$', pano_id):
        return False, 0.0, f"pano_id no-GSV: '{pano_id}'", {}

    loc = metadata.get("location", {})
    pano_lat = loc.get("lat", target_lat)
    pano_lon = loc.get("lng", target_lon)
    dist = haversine_m(pano_lat, pano_lon, target_lat, target_lon)

    if dist > Config.MAX_PANO_DIST_M:
        return False, 0.0, f"Pano lejos: {dist:.0f}m", {}

    date_str = metadata.get("date", "")
    anio_pano = None
    m = re.match(r"(\d{4})", date_str)
    if m:
        anio_pano = int(m.group(1))
        if anio_pano < Config.AÑO_MIN_PANO:
            return False, 0.0, f"Pano viejo: {date_str}", {}

    score_dist  = max(0.0, 1.0 - dist / Config.MAX_PANO_DIST_M)
    score_fecha = 1.0 if (anio_pano and anio_pano >= Config.MIN_FECHA_REC) else 0.7
    score = round(score_dist * score_fecha, 3)

    extras = {
        "pano_lat": pano_lat, "pano_lon": pano_lon,
        "pano_id": pano_id, "date": date_str,
        "dist_m": round(dist, 1), "score_pano": score,
    }
    return True, score, "OK", extras


def is_gray_error_image(img_bytes):
    try:
        img = np.array(Image.open(BytesIO(img_bytes)).convert("L"))
        return float(img.std()) < 8.0
    except Exception:
        return False


def _nombre_archivo(idx, btype, lat, lon, heading, sufijo=""):
    lat_s = f"{lat:.6f}".replace(".", "p").replace("-", "n")
    lon_s = f"{lon:.6f}".replace(".", "p").replace("-", "n")
    hdg_s = f"{heading:.1f}".replace(".", "p")
    btype_safe = "generic" if btype == "yes" else re.sub(r'[^a-z0-9_]', '_', btype.lower())
    return f"ed_{idx:04d}_{btype_safe}_lat{lat_s}_lon{lon_s}_h{hdg_s}_d{sufijo}.jpg"


def download_multi_angulo(target_lat, target_lon, btype, idx, poligono=None):
    """Descarga multi-ángulo de un edificio. Retorna lista de dicts."""
    metadata = get_metadata(target_lat, target_lon)
    ok, score, msg, pano_extra = validar_panorama(metadata, target_lat, target_lon)
    if not ok:
        return [{"ok": False, "info": msg, "idx": idx,
                 "lat": target_lat, "lon": target_lon, "building": btype}]

    pano_lat = pano_extra["pano_lat"]
    pano_lon = pano_extra["pano_lon"]
    dist_m   = pano_extra["dist_m"]

    # Heading: IDECA si disponible, fallback a centroide OSM
    if poligono and poligono.get("heading_fachada"):
        heading_central = poligono["heading_fachada"]
        ancho_m = poligono.get("ancho_m")
        fuente_hdg = "ideca_fachada"
    else:
        heading_central = calculate_heading(pano_lat, pano_lon, target_lat, target_lon)
        ancho_m = None
        fuente_hdg = "centroide_osm"

    pitch = dynamic_pitch(dist_m)
    fov = dynamic_fov(dist_m, ancho_m)

    # Generar ángulos
    if Config.N_ANGULOS == 1:
        angulos = [0]
    else:
        angulos = [0]
        for i in range(1, (Config.N_ANGULOS) // 2 + 1):
            angulos.extend([-Config.DELTA_ANGULO * i, Config.DELTA_ANGULO * i])
        angulos = sorted(angulos[:Config.N_ANGULOS])

    # Descargar cada ángulo
    capturas = []
    for offset in angulos:
        heading = (heading_central + offset) % 360
        params = {
            "size":     Config.SIZE,
            "location": f"{target_lat},{target_lon}",
            "fov":      fov,
            "heading":  round(heading, 1),
            "pitch":    pitch,
            "source":   "outdoor",
            "key":      Config.API_KEY,
        }
        try:
            resp = requests.get(Config.STREETVIEW_URL, params=params, timeout=15)
            if resp.status_code != 200:
                capturas.append({"ok": False, "offset": offset, "info": f"HTTP {resp.status_code}"})
                continue
            if is_gray_error_image(resp.content):
                capturas.append({"ok": False, "offset": offset, "info": "Imagen gris"})
                continue
            score_ang = score * max(0.3, 1.0 - abs(offset) / 90.0)
            capturas.append({
                "ok": True, "offset": offset, "heading": round(heading, 1),
                "score": round(score_ang, 3), "img_bytes": resp.content,
            })
        except Exception as e:
            capturas.append({"ok": False, "offset": offset, "info": str(e)})
        time.sleep(0.05)

    validas = [c for c in capturas if c["ok"]]
    if not validas:
        return [{"ok": False, "info": "Ningún ángulo OK", "idx": idx,
                 "lat": target_lat, "lon": target_lon, "building": btype}]

    # Seleccionar mejor o guardar todas
    a_guardar = validas if Config.GUARDAR_TODOS else [max(validas, key=lambda c: c["score"])]

    resultados = []
    for cap in a_guardar:
        sufijo = f"ang{cap['offset']:+d}" if Config.GUARDAR_TODOS and len(validas) > 1 else ""
        nombre = _nombre_archivo(idx, btype, target_lat, target_lon, cap["heading"], sufijo)
        ruta = Config.DIR_ORIGINALES / nombre
        Image.open(BytesIO(cap["img_bytes"])).save(str(ruta), "JPEG", quality=90)

        resultados.append({
            "ok": True, "info": f"OK ang={cap['offset']:+d}°",
            "idx": idx, "lat": target_lat, "lon": target_lon,
            "building": btype, "filename": nombre,
            "heading": cap["heading"], "pitch": pitch, "fov": fov,
            "dist_m": dist_m, "score": cap["score"],
            "pano_id": pano_extra.get("pano_id", ""),
            "date": pano_extra.get("date", ""),
            "fuente_hdg": fuente_hdg,
            "ancho_fachada_m": ancho_m,
            "ANIO_CONSTR": poligono.get("ANIO_CONSTR") if poligono else None,
            "NUM_PISOS":   poligono.get("NUM_PISOS")   if poligono else None,
            "TIPO_CONST":  poligono.get("TIPO_CONST")   if poligono else None,
        })

    return resultados


def procesar_edificio(row_dict, idx):
    lat = row_dict["lat"]
    lon = row_dict["lon"]
    btype = str(row_dict.get("building", "yes"))
    poligono = consultar_poligono_ideca(lat, lon)
    resultados = download_multi_angulo(lat, lon, btype, idx, poligono)
    for r in resultados:
        r["uso_ideca"] = poligono is not None
    return resultados


# ════════════════════════════════════════════════════════════════
# FASE 1: DESCARGA
# ════════════════════════════════════════════════════════════════

def fase_descarga():
    """Descarga imágenes GSV de Chapinero."""
    if Config.API_KEY.startswith("TU_API"):
        print("\n" + "="*60)
        print("  ERROR: Configura tu API_KEY en Config.API_KEY")
        print("="*60)
        return None

    df = pd.read_csv(Config.CSV_PATH)
    print(f"\n[DESCARGA] CSV cargado: {len(df):,} edificios en Chapinero")

    df_filt = df[~df["building"].isin(Config.EXCLUDE_TYPES)].dropna(subset=["lat", "lon"])
    print(f"[DESCARGA] Disponibles tras filtro: {len(df_filt):,}")

    # ── Excluir edificios ya procesados ─────────────────────────
    report_path = Config.OUTPUT_BASE / "download_report.csv"
    idx_offset = 0
    if report_path.exists():
        df_prev = pd.read_csv(report_path)
        ya_vistos = set(
            zip(df_prev["lat"].round(6), df_prev["lon"].round(6))
        )
        antes = len(df_filt)
        df_filt = df_filt[
            ~df_filt.apply(
                lambda r: (round(r["lat"], 6), round(r["lon"], 6)) in ya_vistos,
                axis=1
            )
        ]
        idx_offset = int(df_prev["idx"].max()) if "idx" in df_prev.columns else len(df_prev)
        print(f"[DESCARGA] Ya procesados: {len(ya_vistos):,} edificios únicos → "
              f"quedan {len(df_filt):,} nuevos (de {antes:,})")
    # ─────────────────────────────────────────────────────────────

    if df_filt.empty:
        print("[DESCARGA] No quedan edificios nuevos para descargar.")
        return None

    n_target = min(len(df_filt), Config.TARGET_COUNT)
    df_target = df_filt.sample(n=n_target).reset_index(drop=True)

    # Estimar costo
    imgs_est = n_target * Config.N_ANGULOS * 1.20
    costo = (imgs_est / 1000) * 7.0
    print(f"\n  Edificios a procesar : {n_target}")
    print(f"  Ángulos por edificio : {Config.N_ANGULOS}")
    print(f"  Costo estimado       : ~${costo:.2f} USD")
    print(f"  Destino              : {Config.DIR_ORIGINALES}")

    resp = input("\n¿Continuar? [S/n] ").strip().lower()
    if resp and resp != "s":
        print("Cancelado.")
        return None

    all_results = []
    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as ex:
        futures = {
            ex.submit(procesar_edificio, row.to_dict(), idx_offset + i + 1): i
            for i, row in df_target.iterrows()
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Descargando"):
            try:
                all_results.extend(fut.result())
            except Exception as e:
                all_results.append({"ok": False, "info": str(e)})

    df_rep = pd.DataFrame(all_results)

    # Agregar al reporte existente (append) en vez de sobreescribir
    if report_path.exists():
        df_rep.to_csv(report_path, mode="a", index=False, header=False, encoding="utf-8-sig")
    else:
        df_rep.to_csv(report_path, index=False, encoding="utf-8-sig")

    ok_n = int(df_rep["ok"].sum())
    fail_n = len(df_rep) - ok_n
    ideca_n = int(df_rep.get("uso_ideca", pd.Series(False)).sum()) if "uso_ideca" in df_rep.columns else 0

    print(f"\n{'='*55}")
    print(f"  Imágenes guardadas  : {ok_n}")
    print(f"  Intentos fallidos   : {fail_n}")
    print(f"  Con polígono IDECA  : {ideca_n}")
    if fail_n > 0 and "info" in df_rep.columns:
        print(f"\n  Motivos de fallo:")
        for motivo, cnt in df_rep[~df_rep["ok"]]["info"].value_counts().head(5).items():
            print(f"    {cnt:3d}x  {motivo}")
    print(f"\n  Reporte: {report_path}")

    return df_rep


# ════════════════════════════════════════════════════════════════
# VALIDACIÓN DE GPU
# ════════════════════════════════════════════════════════════════

def validar_gpu():
    """
    Detecta GPU CUDA disponible y retorna (device, info_dict).
    Ajusta recomendaciones según VRAM disponible.
    """
    info = {"disponible": False, "nombre": "CPU", "vram_total_mb": 0,
            "vram_libre_mb": 0, "compute": "", "usar_amp": False}

    try:
        import torch
    except ImportError:
        print("[GPU] PyTorch no instalado → CPU")
        return torch.device("cpu"), info

    if not torch.cuda.is_available():
        print("[GPU] CUDA no disponible → CPU")
        print("      Verifica: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return torch.device("cpu"), info

    idx = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(idx)
    vram_total = props.total_memory // (1024 ** 2)          # MB
    torch.cuda.empty_cache()
    vram_libre  = (props.total_memory - torch.cuda.memory_allocated(idx)) // (1024 ** 2)
    compute     = f"{props.major}.{props.minor}"
    nombre      = props.name

    info.update({
        "disponible":   True,
        "nombre":       nombre,
        "vram_total_mb": vram_total,
        "vram_libre_mb": vram_libre,
        "compute":      compute,
        "usar_amp":     vram_total < 4096,   # AMP si VRAM < 4 GB
    })

    print(f"\n{'─'*55}")
    print(f"  GPU detectada  : {nombre}")
    print(f"  Compute cap.   : {compute}  (mínimo requerido: 5.0)")
    print(f"  VRAM total     : {vram_total:,} MB")
    print(f"  VRAM libre     : {vram_libre:,} MB")

    if props.major < 5:
        print(f"\n  [WARN] GPU demasiado antigua (compute {compute} < 5.0)")
        print(f"         PyTorch moderno requiere Maxwell o superior.")
        return torch.device("cpu"), info

    if vram_total < 2048:
        print(f"\n  [WARN] VRAM muy limitada ({vram_total} MB).")
        print(f"         SegFormer-B0 puede fallar. Considera reducir SIZE a 320x320.")
    elif vram_total < 4096:
        print(f"  [INFO] VRAM ajustada ({vram_total} MB) → AMP activado (FP16).")
        print(f"         Apto para SegFormer-B0 con imágenes 640×640.")
    else:
        print(f"  [OK]  VRAM suficiente para SegFormer sin restricciones.")

    print(f"{'─'*55}\n")
    return torch.device("cuda", idx), info


# ════════════════════════════════════════════════════════════════
# FASE 2: SEGMENTACIÓN CON SEGFORMER
# ════════════════════════════════════════════════════════════════

def fase_segmentacion(input_dir=None):
    """
    Corre SegFormer sobre las imágenes descargadas.
    Genera:
      - Imagen segmentada (solo fachada, fondo negro, 300×300)
      - Tríptico visual
      - Reporte con proporciones de clase
    """
    try:
        import torch
        import torch.nn.functional as F
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
    except ImportError:
        print("\n[ERROR] Necesitas: pip install torch transformers")
        print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")
        print("  pip install transformers")
        return None

    import cv2

    src_dir = Path(input_dir) if input_dir else Config.DIR_ORIGINALES
    if not src_dir.exists():
        print(f"[ERROR] No existe: {src_dir}")
        return None

    exts = {".jpg", ".jpeg", ".png"}
    imagenes = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in exts)
    print(f"\n[SEGFORMER] {len(imagenes)} imágenes en {src_dir}")

    if not imagenes:
        print("[WARN] No hay imágenes para segmentar.")
        return None

    # Cargar modelo con validación de GPU
    device, gpu_info = validar_gpu()
    usar_amp = gpu_info["usar_amp"] and gpu_info["disponible"]

    print("[SEGFORMER] Cargando modelo SegFormer-B0 (Cityscapes)...")
    processor = SegformerImageProcessor.from_pretrained(Config.SEGFORMER_MODEL)
    model = SegformerForSemanticSegmentation.from_pretrained(Config.SEGFORMER_MODEL)
    model.to(device).eval()
    if usar_amp:
        print(f"[SEGFORMER] Modelo cargado en {device} (FP16/AMP activado para VRAM limitada)")
    else:
        print(f"[SEGFORMER] Modelo cargado en {device}")

    # Mapeo Cityscapes → Custom (0=fondo, 1=cielo, 2=árboles, 3=calle, 4=fachada)
    LOOKUP = np.zeros(256, dtype=np.uint8)
    CS_MAP = {
        0: 3, 1: 3,                            # Road/Sidewalk → Calle
        2: 4, 3: 4, 4: 4,                      # Building/Wall/Fence → Fachada
        5: 0, 6: 0, 7: 0,                      # Postes → Fondo
        8: 2,                                   # Vegetation → Árboles
        9: 3,                                   # Terrain → Calle
        10: 1,                                  # Sky → Cielo
        11: 0, 12: 0,                           # Personas → Fondo
        13: 3, 14: 3, 15: 3, 16: 3, 17: 3, 18: 3  # Vehículos → Calle
    }
    for k, v in CS_MAP.items():
        LOOKUP[k] = v

    # Paleta BGR para visualización
    PALETA = {
        0: [0, 0, 0],         # Fondo → Negro
        1: [255, 191, 0],     # Cielo → Azul
        2: [0, 255, 0],       # Árboles → Verde
        3: [128, 128, 128],   # Calle → Gris
        4: [255, 0, 255],     # Fachada → Fucsia
    }

    resumen = []
    aceptadas = 0
    rechazadas_motivo = {"poca_fachada": 0, "mucha_fachada": 0, "sin_contexto": 0}

    for img_path in tqdm(imagenes, desc="Segmentando"):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inputs = processor(images=img_rgb, return_tensors="pt").to(device)

        with torch.no_grad():
            if usar_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(**inputs).logits
            else:
                logits = model(**inputs).logits
        upsampled = F.interpolate(logits.float(), size=img_bgr.shape[:2],
                                  mode="bilinear", align_corners=False)
        pred = upsampled.argmax(dim=1)[0].cpu().numpy()
        mask = LOOKUP[pred]

        # Proporciones
        total = mask.size
        props = {
            "fondo":   float(np.count_nonzero(mask == 0) / total),
            "cielo":   float(np.count_nonzero(mask == 1) / total),
            "arboles": float(np.count_nonzero(mask == 2) / total),
            "calle":   float(np.count_nonzero(mask == 3) / total),
            "fachada": float(np.count_nonzero(mask == 4) / total),
        }

        # Filtros de calidad
        if props["fachada"] < Config.UMBRAL_MIN_FACHADA:
            rechazadas_motivo["poca_fachada"] += 1
            continue
        if props["fachada"] > Config.UMBRAL_MAX_FACHADA:
            rechazadas_motivo["mucha_fachada"] += 1
            continue
        if (props["cielo"] + props["calle"]) < Config.MIN_CIELO_CALLE:
            rechazadas_motivo["sin_contexto"] += 1
            continue

        # ── Aislar fachada central ──────────────────────────
        mask_fachada = (mask == 4).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_fachada * 255, connectivity=8)

        if num_labels > 1:
            h, w = mask.shape
            centro = (w // 2, h // 2)
            mejor_id = labels[centro[1], centro[0]]
            if mejor_id == 0:
                dist_min = float("inf")
                for i in range(1, num_labels):
                    if stats[i, cv2.CC_STAT_AREA] < 500:
                        continue
                    cx, cy = centroids[i]
                    d = np.sqrt((cx - centro[0])**2 + (cy - centro[1])**2)
                    if d < dist_min:
                        dist_min = d
                        mejor_id = i
            if mejor_id > 0:
                mask_central = np.zeros_like(mask_fachada)
                mask_central[labels == mejor_id] = 1
            else:
                mask_central = mask_fachada
        else:
            mask_central = mask_fachada

        # ── Imagen segmentada para CNN (300×300, fondo negro) ──
        mask_255 = mask_central * 255
        fachada_aislada = cv2.bitwise_and(img_bgr, img_bgr, mask=mask_255)

        # Crop al bounding box + resize
        ys, xs = np.where(mask_central > 0)
        if len(ys) > 0:
            y1, y2 = ys.min(), ys.max()
            x1, x2 = xs.min(), xs.max()
            # Margen del 5%
            h_img, w_img = img_bgr.shape[:2]
            margen_y = int((y2 - y1) * 0.05)
            margen_x = int((x2 - x1) * 0.05)
            y1 = max(0, y1 - margen_y)
            y2 = min(h_img, y2 + margen_y)
            x1 = max(0, x1 - margen_x)
            x2 = min(w_img, x2 + margen_x)
            crop = fachada_aislada[y1:y2, x1:x2]
        else:
            crop = fachada_aislada

        # Resize manteniendo aspecto, padding negro
        th, tw = Config.CNN_SIZE
        h_c, w_c = crop.shape[:2]
        if h_c == 0 or w_c == 0:
            continue
        scale = min(tw / w_c, th / h_c)
        new_w, new_h = int(w_c * scale), int(h_c * scale)
        resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((th, tw, 3), dtype=np.uint8)
        y_off = (th - new_h) // 2
        x_off = (tw - new_w) // 2
        canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        seg_path = Config.DIR_SEGMENTADAS / img_path.name
        cv2.imwrite(str(seg_path), canvas)

        # ── Tríptico visual ─────────────────────────────────
        h_orig, w_orig = img_bgr.shape[:2]

        # Panel 2: segmentación coloreada
        vis_seg = np.zeros_like(img_bgr)
        for cls_id, color in PALETA.items():
            vis_seg[mask == cls_id] = color

        # Panel 3: overlay fucsia
        overlay = img_bgr.copy()
        overlay[mask_central > 0] = [255, 0, 255]
        blend = cv2.addWeighted(img_bgr, 0.6, overlay, 0.4, 0)

        triptico = np.hstack([img_bgr, vis_seg, blend])
        trip_path = Config.DIR_TRIPTICOS / f"trip_{img_path.name}"
        cv2.imwrite(str(trip_path), triptico)

        resumen.append({
            "imagen": img_path.name,
            **{f"prop_{k}": round(v, 4) for k, v in props.items()},
            "frac_central": round(float(mask_central.sum()) / total, 4),
        })
        aceptadas += 1

    # Guardar reporte
    report_path = Config.OUTPUT_BASE / "segmentation_report.csv"
    if resumen:
        pd.DataFrame(resumen).to_csv(report_path, index=False, encoding="utf-8-sig")

    print(f"\n{'='*55}")
    print(f"  Aceptadas            : {aceptadas}")
    print(f"  Rechazadas (poca f.) : {rechazadas_motivo['poca_fachada']}")
    print(f"  Rechazadas (mucha f.): {rechazadas_motivo['mucha_fachada']}")
    print(f"  Rechazadas (contexto): {rechazadas_motivo['sin_contexto']}")
    print(f"\n  Segmentadas en : {Config.DIR_SEGMENTADAS}")
    print(f"  Trípticos en   : {Config.DIR_TRIPTICOS}")
    print(f"  Reporte        : {report_path}")

    return resumen


# ════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline Chapinero: Descarga GSV → SegFormer → Etiquetado")
    parser.add_argument("--mode", choices=["test", "prod", "segment_only"],
                        default="test",
                        help="test=10 edificios, prod=500+, segment_only=solo segmentar")
    parser.add_argument("--target", type=int, default=None,
                        help="Cantidad de edificios a descargar (override)")
    parser.add_argument("--input_dir", type=str, default=None,
                        help="Carpeta de imágenes ya descargadas (para segment_only)")
    parser.add_argument("--api_key", type=str, default=None,
                        help="Google API Key (override)")
    parser.add_argument("--skip_segformer", action="store_true",
                        help="Solo descargar, sin correr SegFormer")
    args = parser.parse_args()

    # Configurar según modo
    if args.mode == "test":
        Config.TARGET_COUNT = 10
        print("\n" + "="*55)
        print("  MODO TEST: 10 edificios de prueba")
        print("  Costo estimado: ~$0.25 USD")
        print("="*55)
    elif args.mode == "prod":
        Config.TARGET_COUNT = args.target or 500
        print("\n" + "="*55)
        print(f"  MODO PRODUCCIÓN: {Config.TARGET_COUNT} edificios")
        print(f"  Costo estimado: ~${(Config.TARGET_COUNT * 3 * 1.2 / 1000) * 7:.2f} USD")
        print("="*55)

    if args.target:
        Config.TARGET_COUNT = args.target
    if args.api_key:
        Config.API_KEY = args.api_key

    # Crear estructura de carpetas
    Config.setup_dirs()

    # Validar GPU al inicio (solo informa, no bloquea)
    if not args.skip_segformer:
        try:
            validar_gpu()
        except Exception:
            pass

    # FASE 1: Descarga
    if args.mode != "segment_only":
        print("\n" + "─"*55)
        print("  FASE 1: Descarga de imágenes GSV")
        print("─"*55)
        fase_descarga()

    # FASE 2: Segmentación
    if not args.skip_segformer:
        print("\n" + "─"*55)
        print("  FASE 2: Segmentación con SegFormer")
        print("─"*55)
        fase_segmentacion(args.input_dir)

    # Resumen final
    n_orig = len(list(Config.DIR_ORIGINALES.glob("*.jpg")))
    n_seg  = len(list(Config.DIR_SEGMENTADAS.glob("*.jpg")))

    print("\n" + "="*55)
    print("  PIPELINE COMPLETO")
    print("="*55)
    print(f"  Originales    : {n_orig} imágenes en {Config.DIR_ORIGINALES}")
    print(f"  Segmentadas   : {n_seg} imágenes en {Config.DIR_SEGMENTADAS}")
    print(f"\n  SIGUIENTE PASO: Etiquetado manual")
    print(f"  ─────────────────────────────────────────────")
    print(f"  1. Abre {Config.DIR_ORIGINALES}")
    print(f"  2. Revisa cada imagen + su tríptico en {Config.DIR_TRIPTICOS}")
    print(f"  3. MUEVE la imagen a la carpeta de su clase:")
    for cls in Config.CLASES:
        print(f"       {Config.OUTPUT_BASE / cls / 'originales'}")
    print(f"  4. Haz lo mismo con la versión segmentada:")
    for cls in Config.CLASES:
        print(f"       {Config.OUTPUT_BASE / cls / 'segmentadas'}")
    print(f"\n  Clases: MAS=Mampostería, DUAL=Sistema dual,")
    print(f"          PC=Pórticos concreto, NC=No clasificable")
    print(f"\n  TIP: Las imágenes malas (lotes vacíos, interiores,")
    print(f"       techos) van a NC/")


if __name__ == "__main__":
    main()