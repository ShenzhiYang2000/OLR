#!/bin/bash

set -x

p_1_to_0=0.0
p_0_to_1=0.0
params3=(0.5)
params4=('ttrl' 'token_entropy' 'seq_entropy' 'self_certainty')

start=5
total_epochs=10
slope_tres=0.025
use_olr=False


ROOT=your_root_path
MODEL_DATABASE=your_path/models/Qwen
MODEL_NAME=Qwen3-4B-Base
SAVE_ROOT=your_save_path
max_response_length=4096


for noise_ratio in "${params3[@]}"; do
    for baseline in "${params4[@]}"; do

        EXP_NAME=DAPO_800_label_flip_${p_1_to_0}_${p_0_to_1}_Qwen3_4B_base_noise_${noise_ratio}_olr_${use_olr}_slope_tres_${slope_tres}_start_${start}_total_epochs_${total_epochs}_${max_response_length}_${baseline}
        CKPT_DIR=$SAVE_ROOT/ckpt/$MODEL_NAME/$EXP_NAME

        echo "Running experiment: $EXP_NAME"

        python3 -m verl.trainer.main_olr \
            algorithm.adv_estimator=grpo \
            data.train_files=/your_path/OLR-main/data/noise_data/DAPO_14k_16384_sampled_800_passrate_0.5_noise_label_${noise_ratio}.parquet \
            data.val_files=/your_path/OLR-main/data/valid_in_training.parquet \
            data.train_batch_size=160 \
            data.max_prompt_length=2048 \
            data.max_response_length=$max_response_length \
            data.filter_overlong_prompts=True \
            data.truncation='error' \
            actor_rollout_ref.model.path=$MODEL_DATABASE/$MODEL_NAME \
            actor_rollout_ref.actor.optim.lr=1e-6 \
            actor_rollout_ref.model.use_remove_padding=True \
            actor_rollout_ref.actor.ppo_mini_batch_size=32 \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
            actor_rollout_ref.actor.use_kl_loss=True \
            actor_rollout_ref.actor.kl_loss_coef=0.001 \
            actor_rollout_ref.actor.kl_loss_type=low_var_kl \
            actor_rollout_ref.actor.entropy_coeff=0 \
            actor_rollout_ref.rollout.temperature=1.0 \
            actor_rollout_ref.model.enable_gradient_checkpointing=True \
            actor_rollout_ref.actor.fsdp_config.param_offload=False \
            actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
            actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
            actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
            actor_rollout_ref.rollout.n=8 \
            actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
            actor_rollout_ref.ref.fsdp_config.param_offload=True \
            algorithm.use_kl_in_reward=False \
            trainer.critic_warmup=1 \
            trainer.logger='["console"]' \
            trainer.project_name=$EXP_NAME \
            trainer.experiment_name=$EXP_NAME \
            +trainer.warm_up=5 \
            +trainer.update_db=True \
            +trainer.use_softmax=False \
            +trainer.softmax_temperature=0.5 \
            +trainer.p_1_to_0=$p_1_to_0 \
            +trainer.p_0_to_1=$p_0_to_1 \
            +trainer.start_select_epoch=$start \
            +trainer.slope_tres=$slope_tres \
            +trainer.use_olr=$use_olr \
            +trainer.baseline=$baseline \
            trainer.train_mode='weak' \
            trainer.output_log_path=/your_path/log/DAPO_800_label_flip_${p_1_to_0}_${p_0_to_1}_Qwen3_4B_base_weak_noise_${noise_ratio}_olr_${use_olr}_slope_tres_${slope_tres}_start_${start}_total_epochs_${total_epochs}_${max_response_length}_${baseline}.log \
            trainer.default_local_dir=$CKPT_DIR \
            trainer.max_actor_ckpt_to_keep=2 \
            trainer.n_gpus_per_node=8 \
            trainer.nnodes=1 \
            trainer.save_freq=100000 \
            trainer.test_freq=5 \
            trainer.total_epochs=$total_epochs
    done
done

