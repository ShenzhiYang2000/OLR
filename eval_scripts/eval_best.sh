ROOT=your_root_path
SAVE_ROOT=your_save_path

GROUP=0

MODEL_PATH=your_path_to_ckpt

EXP_NAME=$(basename "$(dirname "$MODEL_PATH")")

echo $EXP_NAME

TEMPLATE=qwen # llama (if you use llama model)
DATASET=openmath
BASE_MODEL=Qwen3_4B_base

valid_name=(math arc_c gpqa mmlu_pro)


for TASK in "${valid_name[@]}"; do

    DATA=$ROOT/data/valid.$TASK.parquet
    OUTPUT_DIR=$SAVE_ROOT/results/$BASE_MODEL/$DATASET/$EXP_NAME/${TASK}_$GROUP

    if [[ "$GROUP" == "0" ]]; then
        GPUS=0,1,2,3
    else
        GPUS=4,5,6,7
    fi

    mkdir -p $OUTPUT_DIR

    CUDA_VISIBLE_DEVICES=$GPUS python $ROOT/eval_scripts/generate_vllm.py \
      --model_path $MODEL_PATH \
      --input_file $DATA \
      --remove_system True \
      --add_oat_evaluate True \
      --output_file $OUTPUT_DIR/$TASK.jsonl \
      --template $TEMPLATE > $OUTPUT_DIR/$TASK.log

done
