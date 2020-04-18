#!/usr/bin/env bash

# usage
# bash bin/run_experiments.sh {DATA_DIR} {BASE_LOG_DIR} {GPU_ID} {CONFIGS_DIR}

DATA_DIR=$1
BASE_LOG_DIR=$2
GPU_ID=${3:-""}
CONFIGS_DIR=${4:-"model_config"}

for config_path in $(ls -d ${CONFIGS_DIR}/*); do
    echo ">>>>>>>> Training ${config_path}"
    architecture=$(basename ${config_path} .jsonnet)
    log_dir=${BASE_LOG_DIR}/${architecture}
    python train.py \
        --cuda ${GPU_ID} \
        --model_dir ${log_dir} \
        --data_dir ${DATA_DIR} \
        --config ${config_path} \
        --lazy
done

