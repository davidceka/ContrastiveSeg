#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
cd $SCRIPTPATH
cd ../../
. config.profile

SAVE_DIR="${SCRATCH_ROOT}/imaterialist/seg_results/"
DATA_DIR="${DATA_ROOT}/preprocessed/imaterialist"

echo "${DATA_DIR}"
echo "${SAVE_DIR}"

BACKBONE="hrnet48"

CONFIGS="configs/imaterialist/H_48_D_4.json"
CONFIGS_TEST="configs/imaterialist/H_48_D_4_TEST.json"

MODEL_NAME="hrnet_w48_ocr"
LOSS_TYPE="fs_auxce_loss"
CHECKPOINTS_ROOT="/home/david/Documents/repos/progetto-cv/ContrastiveSeg/checkpoints/imaterialist"
CHECKPOINTS_NAME="${MODEL_NAME}_"$2
LOG_FILE="${PROJECT_ROOT}/logs/iMaterialist/${CHECKPOINTS_NAME}.log"
echo "Logging to $LOG_FILE"
mkdir -p `dirname $LOG_FILE`

PRETRAINED_MODEL="/home/david/Documents/repos/progetto-cv/ContrastiveSeg/pretrained_model/hrnetv2_w48_imagenet_pretrained.pth"
MAX_ITERS=200
BATCH_SIZE=2
BASE_LR=0.01

if [ "$1"x == "train"x ]; then
  python -u main.py --configs ${CONFIGS} \
                       --drop_last y \
                       --phase train \
                       --gathered n \
                       --loss_balance y \
                       --log_to_file n \
                       --backbone ${BACKBONE} \
                       --model_name ${MODEL_NAME} \
                       --gpu 0 \
                       --data_dir ${DATA_DIR} \
                       --loss_type ${LOSS_TYPE} \
                       --max_iters ${MAX_ITERS} \
                       --checkpoints_root ${CHECKPOINTS_ROOT} \
                       --checkpoints_name ${CHECKPOINTS_NAME} \
                       --pretrained ${PRETRAINED_MODEL} \
                       --train_batch_size ${BATCH_SIZE} \
                       --distributed \
                       --base_lr ${BASE_LR} \
                       2>&1 | tee ${LOG_FILE}

 elif [ "$1"x == "val"x ]; then
 #  python -u main.py --configs ${CONFIGS} --drop_last y  --data_dir ${DATA_DIR} \
 #                       --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
 #                       --phase test --gpu 0 --resume ${CHECKPOINTS_ROOT}/checkpoints/imaterialist/${CHECKPOINTS_NAME}_latest.pth \
 #                       --loss_type ${LOSS_TYPE} --test_dir ${DATA_DIR}/val/image \
 #                       --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_ms \
 #                       --val_batch_size 1

   #python -m lib.metrics.cityscapes_evaluator --pred_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_ms/label  \
    #                                    --gt_dir ${DATA_DIR}/val/label
    cd lib/metrics
    ${PYTHON} -u ade20k_evaluator.py --configs ../../${CONFIGS_TEST} \
                                   --pred_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_val_ms/label \
                                   --gt_dir ${DATA_DIR}/val/label  

elif [ "$1"x == "segfix"x ]; then
  if [ "$5"x == "test"x ]; then
    DIR=${SAVE_DIR}${CHECKPOINTS_NAME}_test_ss/label
    echo "Applying SegFix for $DIR"
    ${PYTHON} scripts/cityscapes/segfix.py \
      --input $DIR \
      --split test \
      --offset ${DATA_ROOT}/imaterialist/test_offset/semantic/offset_hrnext/
  elif [ "$3"x == "val"x ]; then
    DIR=${SAVE_DIR}${CHECKPOINTS_NAME}_val/label
    echo "Applying SegFix for $DIR"
    ${PYTHON} scripts/cityscapes/segfix.py \
      --input $DIR \
      --split val \
      --offset ${DATA_ROOT}/imaterialist/val/offset_pred/semantic/offset_hrnext/
  fi

elif [ "$1"x == "test"x ]; then
  if [ "$3"x == "ss"x ]; then
    echo "[single scale] test"
    python -u main.py --configs ${CONFIGS} --drop_last y --data_dir ${DATA_DIR} \
                         --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                         --phase test --gpu 0 --resume ${CHECKPOINTS_ROOT}/checkpoints/${CHECKPOINTS_NAME}_latest.pth \
                         --test_dir ${DATA_DIR}/test --log_to_file n \
                         --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_test_ss
  else
    echo "[multiple scale + flip] test"
    python -u main.py --configs ${CONFIGS_TEST} --drop_last y --data_dir ${DATA_DIR} \
                         --backbone ${BACKBONE} --model_name ${MODEL_NAME} --checkpoints_name ${CHECKPOINTS_NAME} \
                         --phase test --gpu 0 --resume ${CHECKPOINTS_ROOT}/checkpoints/imaterialist/${CHECKPOINTS_NAME}_latest.pth \
                         --test_dir ${DATA_DIR}/test --log_to_file n \
                         --out_dir ${SAVE_DIR}${CHECKPOINTS_NAME}_test_ms
  fi


else
  echo "$1"x" is invalid..."
fi