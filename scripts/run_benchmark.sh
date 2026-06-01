#!/usr/bin/env bash
# ============================================================================
# run_benchmark.sh  v3 - Corre las 10 combinaciones (5 modelos x 2 datasets)
# y genera analisis Grad-CAM al final de cada corrida.
#
# CAMBIOS vs v2:
#  - Usa explicitamente el python del venv (.venv/Scripts/python.exe) para
#    asegurar que PyTorch + todas las dependencias estan disponibles.
#  - Apunta a los scripts en src/ (train.py, evaluate.py, gradcam_analysis.py).
#  - Si la primera corrida falla, aborta el benchmark inmediatamente con un
#    mensaje claro (no tiene sentido seguir si train.py no se encuentra o el
#    venv esta mal).
#
# Uso (Git Bash en Windows, desde la raiz del proyecto):
#   bash scripts/run_benchmark.sh datasets_finales runs
#
# Para retomar tras un corte (no re-corre lo terminado):
#   bash scripts/run_benchmark.sh datasets_finales runs --resume
#
# Para cambiar epocas: EPOCHS=50 bash scripts/run_benchmark.sh ...
# ============================================================================

set -u

# ---- FIX UTF-8 (antes de cualquier python) --------------------------------
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export LC_ALL=C.UTF-8 2>/dev/null || true
export LANG=C.UTF-8 2>/dev/null || true
# ---------------------------------------------------------------------------

DATA_ROOT="${1:-datasets_finales}"
OUT_DIR="${2:-runs}"
RESUME_FLAG="${3:-}"

# ---- Python del venv: NO usar el global -----------------------------------
PYTHON_BIN=".venv/Scripts/python.exe"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN=".venv/bin/python"   # fallback Linux/Mac
fi
if [ ! -f "$PYTHON_BIN" ]; then
    echo "ERROR: no encontre el python del venv en .venv/Scripts/python.exe ni .venv/bin/python"
    echo "       Activa el venv o ajusta PYTHON_BIN en este script."
    exit 1
fi
echo "Usando interprete: $PYTHON_BIN"

# ---- Rutas de scripts (todos viven en src/) -------------------------------
TRAIN_PY="src/train.py"
EVAL_PY="src/evaluate.py"
GRADCAM_PY="src/gradcam_analysis.py"

for script in "$TRAIN_PY" "$EVAL_PY" "$GRADCAM_PY"; do
    if [ ! -f "$script" ]; then
        echo "ERROR: no encontre $script (¿estas en la raiz del proyecto?)"
        exit 1
    fi
done

# ---- Parametros de entrenamiento (override con env) -----------------------
EPOCHS="${EPOCHS:-40}"
LR="${LR:-1e-4}"
PATIENCE="${PATIENCE:-8}"

MODELS=(densenet121 resnet50 efficientnetv2_s mobilenetv3_large vit_b_16)
DATASETS=(raw seg)
SEG_DIR="data/piloto/segmentadas"
SKIP_GRADCAM="${SKIP_GRADCAM:-0}"

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/benchmark_log.txt"
echo "============================================================" | tee -a "$LOG"
echo "BENCHMARK iniciado $(date '+%Y-%m-%d %H:%M:%S')"               | tee -a "$LOG"
echo "  DATA_ROOT=$DATA_ROOT  OUT_DIR=$OUT_DIR  RESUME=$RESUME_FLAG" | tee -a "$LOG"
echo "  PYTHON_BIN=$PYTHON_BIN"                                      | tee -a "$LOG"
echo "  EPOCHS=$EPOCHS  PATIENCE=$PATIENCE  LR=$LR"                  | tee -a "$LOG"
echo "  PYTHONIOENCODING=$PYTHONIOENCODING  PYTHONUTF8=$PYTHONUTF8"  | tee -a "$LOG"
echo "============================================================" | tee -a "$LOG"

TOTAL=$(( ${#MODELS[@]} * ${#DATASETS[@]} ))
I=0
N_OK=0
N_FAIL=0
FIRST_RUN=1

for ds in "${DATASETS[@]}"; do
    for m in "${MODELS[@]}"; do
        I=$((I+1))

        CUR_IMG_SIZE=300
        CUR_BATCH=32
        if [ "$m" = "vit_b_16" ]; then
            CUR_IMG_SIZE=224
            CUR_BATCH=16
        fi

        echo                                                              | tee -a "$LOG"
        echo "[$I/$TOTAL] $(date '+%H:%M:%S')  $m  /  D-$ds"               | tee -a "$LOG"
        echo "  img_size=$CUR_IMG_SIZE  batch=$CUR_BATCH"                  | tee -a "$LOG"

        if [ "$RESUME_FLAG" = "--resume" ]; then
            EXISTING=$(ls -1dt "$OUT_DIR"/${m}_${ds}_* 2>/dev/null | head -1 || true)
            if [ -n "$EXISTING" ] && [ -f "$EXISTING/metrics_test.json" ]; then
                echo "  -> ya completado en $EXISTING, saltando"          | tee -a "$LOG"
                N_OK=$((N_OK+1))
                FIRST_RUN=0
                continue
            fi
        fi

        # ---- TRAIN ----
        echo "  >> TRAIN"                                                  | tee -a "$LOG"
        "$PYTHON_BIN" -u "$TRAIN_PY" \
            --data_root "$DATA_ROOT/$ds" \
            --model "$m" \
            --dataset_tag "$ds" \
            --epochs "$EPOCHS" \
            --batch "$CUR_BATCH" \
            --lr "$LR" \
            --patience "$PATIENCE" \
            --img_size "$CUR_IMG_SIZE" \
            --out_dir "$OUT_DIR" 2>&1 | tee -a "$LOG"
        TRAIN_RC=${PIPESTATUS[0]}

        if [ "$TRAIN_RC" -ne 0 ]; then
            echo "  !! TRAIN fallo (rc=$TRAIN_RC)"                         | tee -a "$LOG"
            N_FAIL=$((N_FAIL+1))

            if [ "$FIRST_RUN" = "1" ]; then
                echo                                                       | tee -a "$LOG"
                echo "============================================================" | tee -a "$LOG"
                echo "ABORT: La primera corrida fallo. Tipicamente esto es:"         | tee -a "$LOG"
                echo "  - venv mal activado (no esta torch instalado)"               | tee -a "$LOG"
                echo "  - rutas mal: revisar que src/train.py existe"                | tee -a "$LOG"
                echo "  - datasets_finales/$ds no existe o esta vacio"               | tee -a "$LOG"
                echo "Corrige eso antes de relanzar el benchmark."                   | tee -a "$LOG"
                echo "============================================================" | tee -a "$LOG"
                exit 2
            fi
            continue
        fi
        FIRST_RUN=0

        RUN_DIR=$(ls -1dt "$OUT_DIR"/${m}_${ds}_* 2>/dev/null | head -1)
        if [ -z "$RUN_DIR" ]; then
            echo "  !! no se encontro run_dir para $m/$ds, saltando"      | tee -a "$LOG"
            N_FAIL=$((N_FAIL+1))
            continue
        fi
        echo "  RUN_DIR=$RUN_DIR"                                          | tee -a "$LOG"

        # ---- EVALUATE ----
        echo "  >> EVALUATE"                                               | tee -a "$LOG"
        "$PYTHON_BIN" -u "$EVAL_PY" \
            --run_dir "$RUN_DIR" \
            --data_root "$DATA_ROOT/$ds" \
            --split test 2>&1 | tee -a "$LOG"
        EVAL_RC=${PIPESTATUS[0]}
        if [ "$EVAL_RC" -ne 0 ]; then
            echo "  !! EVALUATE fallo (rc=$EVAL_RC), pero el modelo quedo guardado" | tee -a "$LOG"
        fi

        # ---- GRAD-CAM ----
        if [ "$SKIP_GRADCAM" != "1" ]; then
            echo "  >> GRAD-CAM"                                           | tee -a "$LOG"
            GRADCAM_ARGS=(
                --run_dir "$RUN_DIR"
                --data_root "$DATA_ROOT/$ds"
                --split test
                --max_images 30
                --per_class 10
                --analyze_errors
            )
            if [ -d "$SEG_DIR" ]; then
                GRADCAM_ARGS+=(--seg_dir "$SEG_DIR")
            fi
            "$PYTHON_BIN" -u "$GRADCAM_PY" "${GRADCAM_ARGS[@]}" 2>&1 | tee -a "$LOG"
            GC_RC=${PIPESTATUS[0]}
            if [ "$GC_RC" -ne 0 ]; then
                echo "  !! Grad-CAM fallo, continuando"                    | tee -a "$LOG"
            fi
        fi

        N_OK=$((N_OK+1))
    done
done

echo                                                                       | tee -a "$LOG"
echo "============================================================"        | tee -a "$LOG"
echo "BENCHMARK terminado $(date '+%Y-%m-%d %H:%M:%S')"                   | tee -a "$LOG"
echo "  Exitosos: $N_OK / $TOTAL    Fallidos: $N_FAIL"                    | tee -a "$LOG"
echo "============================================================"        | tee -a "$LOG"
echo                                                                       | tee -a "$LOG"
echo "Siguiente paso:"                                                     | tee -a "$LOG"
echo "  $PYTHON_BIN src/analyze_results.py --runs_dir $OUT_DIR --out_dir $OUT_DIR/_summary" | tee -a "$LOG"
