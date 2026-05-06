#!/usr/bin/env bash
set -xeuo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
EXAMPLE_DIR="/workspace/verl/examples/alpamayo_demo2" # 项目路径文件夹
MODEL_DIR="/workspace/Alpamayo-R1-10B-vlm" # alpamayo的vlm模型文件夹，需要是Qwen3VL架构的
TOKENIZER_DIR="/workspace/Alpamayo-R1-10B-training/" # 需要alpamayo的原config文件中的traj_tokenizer_cfg相关配置
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
    data.train_batch_size=8 \
    data.max_prompt_length=3000 \ # 4096
    data.max_response_length=256 \
    data.filter_overlong_prompts=False \
    data.dataloader_num_workers=0\
    actor_rollout_ref.model.path="$MODEL_DIR" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.name="$ENGINE" \
    actor_rollout_ref.rollout.n=12 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.max_model_len=3300\ # 4352
    actor_rollout_ref.rollout.max_num_seqs=8\
    actor_rollout_ref.rollout.max_num_batched_tokens=26400\ # 34816
    actor_rollout_ref.rollout.gpu_memory_utilization=0.2\
    actor_rollout_ref.rollout.enforce_eager=True\ False
    actor_rollout_ref.rollout.enable_prefix_caching=False\ True
    actor_rollout_ref.rollout.tensor_model_parallel_size=8\
    actor_rollout_ref.rollout.temperature=0.6\
    actor_rollout_ref.rollout.top_p=0.98\
    actor_rollout_ref.rollout.agent.default_agent_loop=alpamayo_prefill_agent\
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$EXAMPLE_DIR/agent_loop.yaml"\
    +actor_rollout_ref.rollout.custom="{eos_token_id:155683}" \
    reward.reward_manager.source=importlib \
    reward.reward_manager.name=AlpamayoSpecialTokenRewardManager \
    reward.reward_manager.module.path="$EXAMPLE_DIR/reward_manager.py" \
    reward.custom_reward_function.path="$EXAMPLE_DIR/reward_fn2.py" \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.model_path="$TOKENIZER_DIR" \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=alpamayo_demo \
    trainer.experiment_name=qwen3_vl_2b_local_demo \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=50 \
    trainer.total_epochs=5 \
    "$@"
