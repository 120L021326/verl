#!/usr/bin/env bash
set -xeuo pipefail

PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
EXAMPLE_DIR=${EXAMPLE_DIR:-"$PROJECT_DIR/examples/alpamayo_demo3"}
MODEL_DIR=${MODEL_DIR:-"/workspace/Alpamayo-R1-10B-vlm"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"/workspace/Alpamayo-R1-10B-training"}
TRAIN_FILE=${TRAIN_FILE:-"$EXAMPLE_DIR/data/train.parquet"}
VAL_FILE=${VAL_FILE:-"$EXAMPLE_DIR/data/val.parquet"}
ENGINE=${ENGINE:-vllm}

# Set REWARD_MODEL_ENABLE=False to disable the internal LRM judge and run
# trajectory-only reward with the same reward function.
REWARD_MODEL_ENABLE=${REWARD_MODEL_ENABLE:-True}
# On a single 8-GPU node, keep the reward model colocated with the actor
# resource pool. Setting this to True asks verl for an extra reward-model
# GPU pool in addition to trainer.n_gpus_per_node.
REWARD_MODEL_ENABLE_RESOURCE_POOL=${REWARD_MODEL_ENABLE_RESOURCE_POOL:-False}
REWARD_MODEL_DIR=${REWARD_MODEL_DIR:-"/share/models/Cosmos-Reason2-8B/"}

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.custom_cls.path="$EXAMPLE_DIR/dataset3.py" \
    data.custom_cls.name=AlpamayoDemoDataset3 \
    data.train_batch_size=8 \
    data.max_prompt_length=3000 \
    data.max_response_length=256 \
    data.filter_overlong_prompts=False \
    data.dataloader_num_workers=0 \
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
    actor_rollout_ref.rollout.max_model_len=3300 \
    actor_rollout_ref.rollout.max_num_seqs=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=26400 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=8 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.98 \
    actor_rollout_ref.rollout.agent.default_agent_loop=alpamayo_prefill_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$EXAMPLE_DIR/agent_loop.yaml" \
    +actor_rollout_ref.rollout.custom="{eos_token_id:155683}" \
    reward.reward_manager.source=importlib \
    reward.reward_manager.name=AlpamayoSpecialTokenRewardManager \
    reward.reward_manager.module.path="$EXAMPLE_DIR/reward_manager.py" \
    reward.custom_reward_function.path="$EXAMPLE_DIR/reward_fn3.py" \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.model_path="$TOKENIZER_DIR" \
    +reward.custom_reward_function.reward_kwargs.traj_weight=1.0 \
    +reward.custom_reward_function.reward_kwargs.reason_weight=1.0 \
    +reward.custom_reward_function.reward_kwargs.enable_reasoning_reward=True \
    reward.reward_model.enable="$REWARD_MODEL_ENABLE" \
    reward.reward_model.enable_resource_pool="$REWARD_MODEL_ENABLE_RESOURCE_POOL" \
    reward.reward_model.n_gpus_per_node=1 \
    reward.reward_model.nnodes=1 \
    reward.reward_model.model_path="$REWARD_MODEL_DIR" \
    reward.reward_model.rollout.name=vllm \
    reward.reward_model.rollout.tensor_model_parallel_size=8 \
    reward.reward_model.rollout.gpu_memory_utilization=0.3 \
    reward.reward_model.rollout.max_model_len=8192 \
    reward.reward_model.rollout.max_num_batched_tokens=8192 \
    reward.reward_model.rollout.max_num_seqs=16 \
    reward.reward_model.rollout.prompt_length=4096 \
    reward.reward_model.rollout.response_length=512 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=alpamayo_demo3 \
    trainer.experiment_name=qwen3_vl_alpamayo_reasoning_reward \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    "$@"
