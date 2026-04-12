#!/bin/bash

set -e  # 任意报错直接退出

# ===============================
# 只需要修改这一行
# ===============================
RUN_PATH=/ossfs/workspace/aitech_aidata/chuwei/ckpt/Qwen3-4B-Base/DAPO_800_label_flip_0.0_0.0_Qwen3_4B_base_strong_noise_0.5_trapo_False_slope_tres_0.025_start_5_total_epochs_10_4096_small_loss_select/global_step_5

# ===============================
# 固定路径（一般不用改）
# ===============================
BASE_DIR="/ossfs/workspace/aitech_aidata/chuwei/ckpt/Qwen3-4B-Base"

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