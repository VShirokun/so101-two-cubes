#!/usr/bin/env bash
# Дообучение GR00T N1.7 на задаче «взять и поднять» (сим или реальный стенд).
# Рецепт проекта (ml/gr00t/README.md): lr 2e-5, batch 8, resize 256, кроп 0.95,
# без letter-box; меняется только датасет, база и бюджет шагов.
#
#   NAME=lift-sim-10k DATA=/srv/data/vshirokun/datasets/gr00t-lift-hd-v21 STEPS=10000 bash train_lift.sh
#   NAME=lift-real-20k DATA=<real v21> STEPS=20000 bash train_lift.sh
#
# Запускать под tmux. Обучение идёт с nice 10: на этой же машине пишется
# реальный датасет, и шине руки нужен CPU (таймаут записи 09.09.2026).
set -uo pipefail
NAME="${NAME:?NAME}"
DATA="${DATA:?DATA}"
STEPS="${STEPS:-20000}"
LR="${LR:-2e-5}"
BASE="${BASE:-/srv/data/vshirokun/lerobot-runs/mix-20k/checkpoint-20000}"   # рекорд 92,17 % в симе
RUNS="${RUNS:-/srv/data/vshirokun/lerobot-runs}"
PYG=/srv/data/vshirokun/groot-venv/bin/python
export LD_LIBRARY_PATH=/srv/data/vshirokun/ffmpeg7-shim:/srv/data/vshirokun/groot-venv/lib/python3.12/site-packages/av.libs
export HF_HOME=/srv/data/vshirokun/hf-cache MUJOCO_GL=egl
# токен HF отозван: бэкбон Cosmos-Reason2 (закрытый) берётся из кэша, обходы gr00t/__init__.py
export HF_HUB_OFFLINE=1 GROOT_PATCH_MISTRAL=1 GROOT_HF_LOCAL_FIRST=1
export PYTORCH_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
mkdir -p "$RUNS"
echo "START $(date -Is) name=$NAME data=$DATA base=$BASE steps=$STEPS lr=$LR"
cd /srv/data/vshirokun/groot-src
nice -n 10 "$PYG" gr00t/experiment/launch_finetune.py \
  --base-model-path "$BASE" --dataset-path "$DATA" \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/SO100/so100_config.py \
  --num-gpus 1 --output-dir "$RUNS/$NAME" \
  --max-steps "$STEPS" --save-steps "$STEPS" --save-total-limit 2 \
  --learning-rate "$LR" \
  --shortest-image-edge 256 --crop-fraction 0.95 --no-letter-box \
  --global-batch-size 8 --dataloader-num-workers 4 \
  2>&1 | tee "$RUNS/$NAME.log" | grep -E "train_runtime|Error|error|Traceback" | tail -3
echo "EXIT:${PIPESTATUS[0]} $(date -Is)"
echo "checkpoint: $RUNS/$NAME/checkpoint-$STEPS"
