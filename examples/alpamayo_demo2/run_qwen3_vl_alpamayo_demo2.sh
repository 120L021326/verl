#!/usr/bin/env bash
set -xeuo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
EXAMPLE_DIR="/workspace/verl/examples/alpamayo_demo"
MODEL_DIR="/workspace/Alpamayo-R1-10B-vlm"
TOKENIZER_DIR="/workspace/Alpamayo-R1-10B-training/"
TRAIN_FILE="/workspace/alpamayo_metadata/train.parquet"
VAL_FILE="/workspace/alpamayo_metadata/val.parquet"
ENGINE=${ENGINE:-vllm}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.custom_cls.path="$EXAMPLE_DIR/dataset2.py" \
    data.custom_cls.name=AlpamayoDemoDataset2 \
    data.train_batch_size=4 \
    data.max_prompt_length=3000 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=False \
    data.dataloader_num_workers=0\
    actor_rollout_ref.model.path="$MODEL_DIR" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.name="$ENGINE" \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.max_model_len=3300\
    actor_rollout_ref.rollout.max_num_seqs=8\
    actor_rollout_ref.rollout.max_num_batched_tokens=3300\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2\
    actor_rollout_ref.rollout.enforce_eager=True\
    actor_rollout_ref.rollout.enable_prefix_caching=False\
    actor_rollout_ref.rollout.tensor_model_parallel_size=8\
    +actor_rollout_ref.rollout.custom="{eos_token_id:155683}" \
    reward.reward_manager.source=importlib \
    reward.reward_manager.name=AlpamayoSpecialTokenRewardManager \
    reward.reward_manager.module.path="$EXAMPLE_DIR/reward_manager.py" \
    reward.custom_reward_function.path="$EXAMPLE_DIR/reward_fn2.py" \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.model_path="$TOKENIZER_DIR" \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=alpamayo_demo \
    trainer.experiment_name=qwen3_vl_2b_local_demo \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.total_epochs=1 \
    "$@"
