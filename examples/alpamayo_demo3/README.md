# Alpamayo Demo3

`alpamayo_demo3` 基于 `alpamayo_demo2`，在原有轨迹 reward 的基础上加入
Alpamayo 风格的 Reasoning Quality Reward。

核心思路是：模型 rollout 先生成 `<|cot_start|>...<|cot_end|>` 中的 reasoning
trace，再生成 `<|traj_future_start|>...<|traj_future_end|>` 中的未来轨迹 token。
`reward_fn3.py` 会同时计算轨迹 reward 和 reasoning reward。reasoning reward
通过 verl 内部启动的 generative reward model / LRM judge 完成，不需要用户手动在
外部启动 judge 服务。

## 与 Demo2（即无 Reward Model 的 demo） 的区别

- `dataset3.py` 支持从 `ground_truth`、`coc`、`cot`、`gt_cot`、`reasoning`
  或 `extra_info` 中的同名字段读取 GT reasoning。
- `reward_fn3.py` 会抽取 `<|cot_start|>` 和 `<|cot_end|>` 之间的预测 reasoning，
  调用 verl 内部 reward model judge，按照 0-5 分 rubric 对预测 reasoning 和
  GT reasoning 做一致性评分，并将分数归一化到 0-1。
- `run_qwen3_vl_alpamayo_demo3.sh` 启用了 `reward.reward_model`。verl 会启动一个
  vLLM reward model server，并把内部 `reward_router_address` 传给自定义 reward
  function。

## 数据要求

Reasoning Quality Reward 需要每条样本提供 ground-truth CoC / CoT。可使用以下任一字段：

```text
ground_truth
coc
cot
gt_cot
reasoning
extra_info.ground_truth
extra_info.coc
extra_info.cot
extra_info.gt_cot
extra_info.reasoning
```

当前 parquet 可以类似：

```python
{
    "clip_id": "03a83788-ce57-4bec-bf99-e6e82f64227a",
    "t0_us": 7156134,
    "coc": "Yield to crossing pedestrians at the crosswalk.",
    "ground_truth": "Yield to crossing pedestrians at the crosswalk.",
    "data_source": "alpamayo_physical_ai_av_coc",
    "event_cluster": "PEDESTRIAN_DENSITY_OR_CLOSE_PROXIMITY",
    "feature": "camera_front_wide_120fov",
    "idx": 0,
}
```

`clip_id` 和 `t0_us` 用于从本地 Physical AI AV 数据集中加载图像、历史轨迹和未来轨迹。
`ground_truth` / `coc` 用于 Reasoning Quality Reward 的 GT reasoning。

## 从 OOD Reasoning 数据构建 Metadata

可以使用 `scripts/build_metadata.sh` 从本地 Physical AI AV 数据集和 OOD reasoning
parquet 生成 Demo3 的 train / val metadata。

reasoning parquet 需要包含：

```text
clip_id
events
```

`clip_id` 可以是普通列，也可以是 DataFrame 的 index，index 名称为 `clip_id`。

`events` 应该是 JSON event list，每个 event 可以包含：

```text
event_start_frame
event_start_timestamp
coc
```

脚本会根据本地 PAI `clip_index.parquet` 按 chunk 过滤 clip，把每个 event 展开成一条
metadata，保留 `coc` 字段，并写出：

```text
examples/alpamayo_demo3/data/train.parquet
examples/alpamayo_demo3/data/val.parquet
```

基础用法：

```bash
cd examples/alpamayo_demo3

./scripts/build_metadata.sh \
  --data-dir /share/datasets/PhysicalAI-Autonomous-Vehicles \
  --coc-parquet /path/to/ood_reasoning.parquet \
  --chunk-ids 3116-3120 \
  --force-rebuild
```

在 chunk 过滤后随机抽样固定行数：

```bash
./scripts/build_metadata.sh \
  --data-dir /share/datasets/PhysicalAI-Autonomous-Vehicles \
  --coc-parquet /path/to/ood_reasoning.parquet \
  --chunk-ids 3116-3120 \
  --num-samples 320 \
  --random-seed 42 \
  --force-rebuild
```

默认情况下，`t0_us` 来自每个 event 的 `event_start_timestamp`。如果要从
`event_start_frame` 推导 `t0_us`：

```bash
./scripts/build_metadata.sh \
  --data-dir /share/datasets/PhysicalAI-Autonomous-Vehicles \
  --coc-parquet /path/to/ood_reasoning.parquet \
  --chunk-ids 3116-3120 \
  --t0-source event_frame \
  --event-frame-rate 10.0 \
  --force-rebuild
```

如果希望采样事件开始前一小段时间的图像，可以加负 offset。例如把 `t0_us` 设为事件开始前
0.3 秒：

```bash
./scripts/build_metadata.sh \
  --data-dir /share/datasets/PhysicalAI-Autonomous-Vehicles \
  --coc-parquet /path/to/ood_reasoning.parquet \
  --chunk-ids 3116-3120 \
  --event-time-offset-us -300000 \
  --force-rebuild
```

常用参数：

- `--metadata-output-dir`：`train.parquet` 和 `val.parquet` 输出目录。
- `--events-column`：event list 所在列名，默认 `events`。
- `--val-ratio`：验证集比例，默认 `0.1`。
- `--min-t0-us`：丢弃过早样本，避免不满足默认 1.6 秒 history window。
- `--force-rebuild`：覆盖已有 train / val metadata。

## 运行

按需设置 actor 模型、轨迹 tokenizer config、数据文件和 reward judge 模型：

```bash
export MODEL_DIR=/workspace/Alpamayo-R1-10B-vlm
export TOKENIZER_DIR=/workspace/Alpamayo-R1-10B-training
export TRAIN_FILE=/workspace/alpamayo_metadata_with_cot/train.parquet
export VAL_FILE=/workspace/alpamayo_metadata_with_cot/val.parquet
export REWARD_MODEL_DIR=/share/models/Cosmos-Reason2-8B

bash examples/alpamayo_demo3/run_qwen3_vl_alpamayo_demo3.sh
```

如果只想调试 trajectory reward，可以关闭内部 reward model：

```bash
REWARD_MODEL_ENABLE=False \
bash examples/alpamayo_demo3/run_qwen3_vl_alpamayo_demo3.sh \
  +reward.custom_reward_function.reward_kwargs.enable_reasoning_reward=False
```

默认脚本在单机 8 卡上使用 colocate reward model：

```bash
reward.reward_model.enable=True
reward.reward_model.enable_resource_pool=False
```

这样不会额外申请第 9 张 GPU。如果有独立 reward model GPU 资源池，可以显式设置：

```bash
REWARD_MODEL_ENABLE_RESOURCE_POOL=True
```

## Reward 计算

默认最终 reward 为：

```text
score = 1.0 * trajectory_reward + 1.0 * reasoning_quality_reward
```

可以通过以下参数调整权重：

```bash
+reward.custom_reward_function.reward_kwargs.traj_weight=1.0
+reward.custom_reward_function.reward_kwargs.reason_weight=1.0
```

其中：

```text
reasoning_quality_reward = judge_score / 5.0
```

`judge_score` 由内部 LRM judge 按 0-5 分 rubric 输出。

## 已知问题与 ray_trainer.py 修改原因

Demo3 启用内部 generative reward model 后，和 Demo2（即没有用到reward model的demo） 最大的执行路径差异是：

```bash
reward.reward_model.enable=True
reward.reward_model.enable_resource_pool=False
```

这表示 reward model 与 actor/rollout 共享同一个 GPU resource pool。verl 会先完成 rollout，
然后在 trainer 侧调用 colocate reward model 计算 reward。

在这种路径下，agent loop 没有拿到 `reward_loop_worker_handles`，因此它会把输入 batch 的
`non_tensor_batch` 原样复制到 rollout 输出 batch。相关逻辑在
`verl/experimental/agent_loop/agent_loop.py` 中：

```python
if self.reward_loop_worker_handles is None and input_non_tensor_batch:
    non_tensor_batch.update(input_non_tensor_batch)
```

Alpamayo Demo 的 `extra_info` 中包含轨迹 Tensor：

```text
ego_history_xyz
ego_history_rot
ego_future_xyz
ego_future_rot
```

这些 Tensor 是 trajectory reward 所需的参考数据。问题出现在 trainer 后续会执行
`DataProto.union()`：

```python
test_batch = test_batch.union(test_output_gen_batch)
batch = batch.union(gen_batch_output)
```

`DataProto.union()` 在发现两个 batch 都有同名 `non_tensor_batch` key 时，会检查它们是否
完全相等。对于 `extra_info` 这种 object array，比较会递归进入 dict，最终执行：

```python
tensor_a == tensor_b
```

多元素 Tensor 的比较结果仍然是 Tensor，不能作为 Python bool，因此报错：

```text
RuntimeError: Boolean value of Tensor with more than one value is ambiguous
```

Demo2 通常不会遇到这个问题，是因为 Demo2 没有启用内部 reward model：

```bash
reward.reward_model.enable=False
```

此时 reward 可以在 agent loop 阶段计算，`reward_loop_worker_handles` 不为空，rollout 输出 batch
不会原样复制输入的 `extra_info`，因此不会触发 `extra_info` 中 Tensor 的深度比较。

为了解决 Demo3 的 colocate reward model 路径问题，修改了
`verl/trainer/ppo/ray_trainer.py`：

```python
def _drop_conflicting_tensor_non_tensor_batch(base: DataProto, other: DataProto) -> DataProto:
    ...
```

该函数只删除 rollout 输出 batch 中满足以下条件的 `non_tensor_batch` 字段：

1. 这个 key 同时存在于原始 batch 和 rollout 输出 batch；
2. 原始值或输出值中递归包含 `torch.Tensor`。

也就是说，它会删除输出 batch 中重复且含 Tensor 的 `extra_info`，避免 union 比较 Tensor；
但会保留只存在于 rollout 输出 batch 的字段，例如 `multi_modal_inputs`。`multi_modal_inputs`
虽然也可能含 Tensor，但原始 batch 没有同名 key，`DataProto.union()` 不会比较它，只会把它加入
合并后的 batch，后续训练仍然可以使用它。

当前修改点：

```python
test_output_gen_batch = _drop_conflicting_tensor_non_tensor_batch(test_batch, test_output_gen_batch)
test_batch = test_batch.union(test_output_gen_batch)
```

以及：

```python
gen_batch_output = _drop_conflicting_tensor_non_tensor_batch(batch, gen_batch_output)
batch = batch.union(gen_batch_output)
```

这个改动的目标是最小化影响：保留原始 batch 中的 Alpamayo 参考轨迹 Tensor，保留 rollout 输出中
训练需要的多模态字段，只移除会在 `union()` 中触发 Tensor equality 问题的重复字段。
