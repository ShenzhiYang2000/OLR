#!/bin/bash

set -e  # 任意报错直接退出

# ===============================
# 只需要修改这一行
# ===============================
RUN_PATH=/ossfs/workspace/aitech_aidata/chuwei/ckpt/R1-Distill-Llama-8B/DAPO_800_label_flip_0.0_0.0_R1-Distill-Llama-8B_weak_noise_0.5_trapo_True_slope_tres_0.05_start_4_total_epochs_10_4096/global_step_50

# ===============================
# 固定路径（一般不用改）
# ===============================
BASE_DIR="/ossfs/workspace/aitech_aidata/chuwei/ckpt/R1-Distill-Llama-8B"

# LOCAL_DIR="${BASE_DIR}/${RUN_PATH}/actor"
# TARGET_DIR="${BASE_DIR}/${RUN_PATH}_merge_hf"

LOCAL_DIR="${RUN_PATH}/actor"
TARGET_DIR="${RUN_PATH}_merge_hf"

echo "======================================"
echo "Merging model..."
echo "Source: ${LOCAL_DIR}"
echo "Target: ${TARGET_DIR}"
echo "======================================"

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${LOCAL_DIR}" \
    --target_dir "${TARGET_DIR}"

echo "======================================"
echo "Merge completed successfully."
echo "Saved to: ${TARGET_DIR}"
echo "======================================"

# ===============================
# 删除 LOCAL_DIR
# ===============================
if [ -d "${LOCAL_DIR}" ]; then
    echo "Deleting ${LOCAL_DIR} ..."
    rm -rf "${LOCAL_DIR}"
    echo "Deleted ${LOCAL_DIR}"
else
    echo "Warning: ${LOCAL_DIR} not found, skip deletion."
fi