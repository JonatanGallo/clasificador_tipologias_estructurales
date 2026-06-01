# Clasificación de tipologías estructurales de edificaciones — Chapinero, Bogotá

Pipeline semi-automatizado para estimar la **tipología estructural** de edificaciones
(mampostería *MAS*, pórticos de concreto *PC*, sistema dual *DUAL*) a partir de
imágenes de fachada de **Google Street View (GSV)**, huellas de **OpenStreetMap**,
**segmentación semántica** con SegFormer y **clasificación con deep learning**.
El proyecto incluye un análisis de interpretabilidad con **Grad-CAM** orientado a la
modelación de exposición sísmica urbana.

> El etiquetado es **manual** (asistido por un experto), por lo que el sistema es
> **semi-automático**: todo lo demás —adquisición, segmentación, construcción del
> dataset, entrenamiento, evaluación e interpretabilidad— está automatizado.

Trabajo de grado de la Maestría en Inteligencia Artificial, Pontificia Universidad
Javeriana.

---

## Pipeline (de extremo a extremo)

1. **Extracción de coordenadas** — huellas de edificios desde OSM (`osmnx`), filtrando
   tipos no habitacionales.
2. **Adquisición GSV** — para cada coordenada se valida el panorama (metadatos,
   proximidad, vigencia), se calcula el *heading* (catastro IDECA si está disponible,
   si no el centroide OSM) y se descarga la imagen.
3. **Segmentación semántica** — SegFormer-B0 (Cityscapes) aísla la fachada y produce
   las máscaras usadas tanto para el dataset segmentado como para el análisis Grad-CAM.
4. **Etiquetado manual asistido** — herramienta de escritorio (Tkinter) con vista
   original + segmentada y atajos de teclado (MAS / PC / DUAL / NC).
5. **Construcción del dataset** — genera **D-RAW** (imagen completa) y **D-SEG**
   (fachada aislada) con **partición espacial por grilla** para evitar fuga geográfica.
6. **Entrenamiento y benchmark** — 5 arquitecturas × 2 datasets (10 corridas).
7. **Evaluación e interpretabilidad** — métricas por clase, matrices de confusión y
   análisis cuantitativo de zonas Grad-CAM (% de activación sobre fachada).

Arquitecturas evaluadas: **DenseNet121, ResNet50, EfficientNetV2-S,
MobileNetV3-Large, ViT-B/16**.

---

## Estructura del repositorio

```
.
├── README.md
├── requirements.txt
├── .env.example                 # plantilla para la API key (sin la clave real)
├── .gitignore
├── Pipeline_chapinero.py        # Fases 1–3: adquisición GSV + segmentación
├── dataset_builder.py           # Fase 5: D-RAW / D-SEG + partición espacial
├── src/
│   ├── etiquetador.py           # Fase 4: herramienta de etiquetado (Tkinter)
│   ├── train.py                 # entrenamiento parametrizable
│   ├── evaluate.py              # evaluación (métricas, matriz de confusión)
│   └── gradcam_analysis.py      # interpretabilidad + análisis de zonas
├── scripts/
│   └── run_benchmark.sh         # 10 corridas (5 modelos × 2 datasets)
├── datasets_finales/            # SOLO los CSV de splits (las imágenes NO se versionan)
│   ├── raw/{train,val,test}.csv, split_report.txt
│   └── seg/{train,val,test}.csv, split_report.txt
└── benchmark_summary.csv        # resultados consolidados
```

> **No se versionan**: imágenes (`.jpg`), pesos de modelos (`.pt`), la carpeta del
> artículo (`files/`, LaTeX), las figuras generadas ni los scripts de figuras.
> Las imágenes de GSV no pueden redistribuirse por las condiciones de uso de Google;
> el repositorio conserva las coordenadas y las etiquetas para poder regenerarlas.

---

## Configuración

```bash
# 1. Entorno virtual
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# 2. Dependencias
pip install -r requirements.txt

# 3. API key de Google Street View (NUNCA hardcodear en el código)
cp .env.example .env            # y edita .env con tu clave real
```

El código lee la clave desde la variable de entorno `GSV_API_KEY` (vía `.env`).
Restringe la clave en Google Cloud Console a *Street View Static API* y
*Street View Static Metadata*.

---

## Uso

```bash
# Fases 1–3 — adquisición + segmentación (modo prueba con pocas imágenes)
python Pipeline_chapinero.py --mode test --target 10
# Producción:
python Pipeline_chapinero.py --mode prod --target 600

# Fase 4 — etiquetado manual
python src/etiquetador.py

# Fase 5 — construcción del dataset con partición espacial
python dataset_builder.py \
    --dataset_dir data/piloto \
    --labels_csv  data/piloto/labels.csv \
    --out_dir     datasets_finales \
    --grid_size 0.005 --val_frac 0.15 --test_frac 0.15 \
    --img_size 300 --seed 42

# Fase 6 — entrenamiento de un modelo
python src/train.py --data_root datasets_finales/raw --model densenet121 \
    --dataset_tag raw --epochs 40 --batch 32 --lr 1e-4 \
    --img_size 300 --patience 5 --seed 42 --out_dir runs/

# Benchmark completo (10 corridas + Grad-CAM)
bash scripts/run_benchmark.sh datasets_finales runs

# Fase 7 — evaluación de una corrida
python src/evaluate.py --run_dir runs/<corrida> --data_root datasets_finales/raw --split test
```

---

## Esquema de clases

| Clase | Sistema estructural                     | Señales visuales típicas                          |
|-------|-----------------------------------------|---------------------------------------------------|
| MAS   | Mampostería (no reforzada / confinada)  | Ladrillo o pañete, 1–3 pisos, vanos irregulares   |
| PC    | Pórticos de concreto reforzado          | Mediana altura, ventanería repetitiva             |
| DUAL  | Pórtico + muros de cortante             | Gran altura, fachada rígida, paneles              |
| NC    | No clasificable (solo etiquetado)       | Lotes vacíos, oclusión severa — se excluye         |

La distinción PC vs. DUAL depende de muros internos no visibles desde la fachada,
lo que impone un límite observacional al enfoque.

---

## Fuentes de datos

- **OpenStreetMap** — huellas de edificios (`osmnx`).
- **Google Street View Static API** — imágenes de fachada.
- **IDECA (Catastro de Bogotá)** — polígonos para el cálculo de *heading*.
- **SegFormer-B0** (`nvidia/segformer-b0-finetuned-cityscapes-512-1024`) — segmentación.

---

## Cita

Si usas este trabajo, cita el artículo asociado (en revisión):
Gallo Martínez, J. A.; Villalba Morales, J. D.; Caicedo Dorado, A.
*Clasificación automática de tipologías estructurales de edificaciones mediante deep
learning e imágenes de Google Street View: un estudio interpretable en Chapinero, Bogotá.*

---

## Licencia y datos

Código bajo la licencia que definas (sugerencia: MIT para el código).
Las imágenes de Google Street View **no se incluyen ni se redistribuyen**;
su uso está sujeto a las condiciones de Google Maps Platform.
