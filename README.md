# verl 代码学习笔记
### 数据划分配置
按 `examples/alpamayo_demo/run_qwen3_vl_2b_demo.sh` 里的关键配置：

```text
data.train_batch_size = 2
actor_rollout_ref.rollout.n = 2
actor_rollout_ref.actor.ppo_mini_batch_size = 2
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu = 1
trainer.n_gpus_per_node = 4   
```

按 demo 配置、数据集 8 条、4 GPU 时，流程是：

1. **DataLoader 取原始样本**
   - `data.train_batch_size=2` 表示每个训练 step 从训练集取 2 条原始数据。
   - 8 条数据、`drop_last=True`，每 epoch 有 `8 / 2 = 4` 个训练 step。
   - demo 没设置 `data.shuffle`，默认是 `True`，所以顺序会被随机采样。

2. **每条样本生成多个 rollout**
   - `rollout.n=2` 表示每条 prompt 生成 2 个 response。
   - 所以一个 step 中：
     - 原始 prompt：2 条
     - rollout 后序列：`2 * 2 = 4` 条
   - 形态类似：
     ```text
     sample0 -> response0, response1
     sample1 -> response0, response1
     ```

3. **rollout 分到 GPU**
   - 4 条 rollout 序列切给 4 GPU。
   - 每张 GPU 处理 1 条生成序列。

4. **reward / advantage**
   - reward 对 4 条 response 分别打分。
   - GRPO 会按原始 prompt 分组：每组 2 条 response。
   - 也就是 sample0 的 2 个 response 互相比较，sample1 的 2 个 response 互相比较，然后算 advantage。

5. **actor PPO update**
   - `ppo_mini_batch_size=2` 是按“原始 prompt 数”理解；代码里会乘 `rollout.n`。
   - 所以 PPO 全局 mini-batch 实际是：
     ```text
     2 * 2 = 4 条 rollout 序列
     ```
   - 当前 step 正好也是 4 条 rollout 序列，所以只有 1 个 PPO mini-batch。
   - 4 GPU 下，每 GPU 拿 1 条序列。

6. **micro batch**
   - `ppo_micro_batch_size_per_gpu=1` 表示每张 GPU 上一次 forward/backward 处理 1 条序列。
   - 因为每 GPU mini-batch 也是 1，所以没有额外梯度累积。
   - 如果每 GPU mini-batch 是 2，micro=1，就会拆成 2 次 micro forward/backward，再做一次 optimizer step。

如果要用 **8 GPU + 8 条数据**，建议至少改成下面这种之一：

```text
data.train_batch_size=4
actor_rollout_ref.actor.ppo_mini_batch_size=4
actor_rollout_ref.rollout.n=2
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
trainer.n_gpus_per_node=8
```

这时每 step：

```text
4 条原始样本
-> 每条生成 2 个 response
-> 8 条 rollout 序列
-> 8 GPU 每张 1 条
-> 1 个 PPO mini-batch
-> 每 GPU micro batch = 1
```

8 条数据一共跑 2 个 training step。

如果想一整个 epoch 只跑 1 step，可以用：

```text
data.train_batch_size=8
actor_rollout_ref.actor.ppo_mini_batch_size=8
```

这时：

```text
8 条原始样本
-> 16 条 rollout 序列
-> 8 GPU 每张 2 条
-> 每 GPU 按 micro=1 拆成 2 次 backward 累积
-> 1 次 optimizer step
```

核心记法是：

```text
每步原始样本数 = data.train_batch_size
每步训练序列数 = data.train_batch_size * rollout.n
PPO 全局 mini-batch 序列数 = ppo_mini_batch_size * rollout.n
每 GPU micro batch = ppo_micro_batch_size_per_gpu
```

对 8 GPU，至少要保证：

```text
data.train_batch_size * rollout.n >= 8
ppo_mini_batch_size * rollout.n >= 8
```

并且通常要能被 8 整除。

---

**actor 的 optimizer step 是按 PPO mini-batch 触发的**，不是按 micro-batch，也不是等整个 epoch。

在当前 `verl.trainer.ppo.ray_trainer` 流程里，一个训练 step 大致是：

```text
取 data.train_batch_size 条 prompt
-> rollout 生成 responses
-> reward
-> old_log_probs
-> ref_log_prob
-> advantage
-> update_actor
-> sync actor weights to rollout
-> 下一个训练 step
```

进入 `update_actor` 后，actor 内部再这样切：

```text
整个训练 batch
-> 按 ppo_mini_batch_size 切成 PPO mini-batches
-> 每个 mini-batch 再按 ppo_micro_batch_size_per_gpu 切成 micro-batches
-> micro-batch 只做 forward/backward 累积梯度
-> 一个 PPO mini-batch 的所有 micro-batch 跑完后，调用一次 optimizer.step()
```

所以关系是：

```text
micro-batch：显存切分单位，只累积梯度
PPO mini-batch：一次 actor 参数更新单位
train batch：一次 rollout + reward + advantage 的数据来源
```

按你 demo 配置：

```text
data.train_batch_size = 2
rollout.n = 2
ppo_mini_batch_size = 2
ppo_micro_batch_size_per_gpu = 1
ppo_epochs = 1  # 默认
```

每个训练 step：

```text
2 条原始样本
-> 4 条 rollout 序列
-> PPO mini-batch 也是 2 条原始样本 * 2 rollout = 4 条序列
-> 只有 1 个 PPO mini-batch
-> actor optimizer.step() 1 次
-> 然后同步给 rollout
```

如果你改成比如：

```text
data.train_batch_size = 8
rollout.n = 2
ppo_mini_batch_size = 4
ppo_micro_batch_size_per_gpu = 1
```

那每个训练 step 是：

```text
8 条原始样本
-> 16 条 rollout 序列
-> 每个 PPO mini-batch = 4 * 2 = 8 条 rollout 序列
-> 一共 2 个 PPO mini-batch
-> actor optimizer.step() 2 次
-> 这 2 次都完成后，才同步给 rollout
```

**rollout 权重同步时机**：不是每个 PPO mini-batch 后同步，而是 `_update_actor(batch)` 整个返回之后同步一次。也就是说，一个 train batch 内如果有多个 PPO mini-batch，actor 会连续更新多次，但 rollout 仍然用旧权重，直到这个训练 step 的 actor update 全部结束，才执行：

```text
checkpoint_manager.update_weights(global_steps)
```

另外训练刚开始也会先同步一次，用来确保 rollout 侧拿到初始 actor 权重。demo 里 `critic_warmup=0`，所以正常每个训练 step 都是：

```text
rollout 用当前已同步权重生成
-> actor 用这批 rollout 更新
-> 更新完成后同步新 actor 权重给 rollout
-> 下一 step 的 rollout 使用新权重
```

---

`ppo_epochs` 表示：**同一批 rollout 数据被 actor 重复训练多少轮**。

在 verl 里，一次训练 step 会先生成一批数据：

```text
data.train_batch_size 条 prompt
-> 每条生成 rollout.n 个 response
-> 算 reward / advantage / old_log_probs
```

然后进入 actor update。`ppo_epochs` 控制这批已经生成好的数据会被重复用于 PPO 更新几遍：

```text
for epoch in range(ppo_epochs):
    for mini_batch in PPO mini-batches:
        forward/backward
        optimizer.step()
```

所以：

```text
actor optimizer step 次数
= ppo_epochs * (train_batch_size / ppo_mini_batch_size)
```

这里的 `train_batch_size` 和 `ppo_mini_batch_size` 都按“原始 prompt 数”理解；代码里会一起乘 `rollout.n`，比例不变。

你的 demo 默认没有显式设置，所以用配置默认值：

```text
actor_rollout_ref.actor.ppo_epochs = 1
```

也就是每批 rollout 数据只训练一遍。

举例：

```text
data.train_batch_size = 8
rollout.n = 2
ppo_mini_batch_size = 4
ppo_epochs = 3
```

流程是：

```text
一次生成 8 * 2 = 16 条 rollout 序列
每个 PPO mini-batch = 4 * 2 = 8 条序列
每轮有 2 个 mini-batch
ppo_epochs=3，所以同一批 16 条序列会被重复训练 3 轮
actor optimizer.step() = 3 * 2 = 6 次
```

注意：`ppo_epochs` 不会重新 rollout。它只是复用同一批旧 response、reward、advantage 多训练几遍；全部 PPO epoch 完成后，才把 actor 新权重同步给 rollout，用于下一批生成。

### verl 中的 worker 设计
[verl.single_controller 的设计](https://verl.org.cn/en/latest/single_controller.html)
single_controller 的目标，是让你写分布式 RLHF / PPO 时，代码看起来尽量像单进程脚本。

也就是你本来写：
```
rollout = Rollout()
rollout.generate_sequences(batch)
```
迁移到分布式后，尽量还是写成：
```
rollout = RayWorkerGroup(...)
rollout.generate_sequences(batch)
```
也就是说，verl会自行处理分发、整合等分布式细节，这些细节通过`MAGIC_ATTR`定义。
`MAGIC_ATTR`中`dispatch_mode / execute_mode / blocking`可以找到对应的 `dispatch_fn / collect_fn / execute_fn`：
- dispatch_fn 决定怎么分输入
- execute_fn 决定怎么真正发起远程执行
- collect_fn 决定怎么收结果

### 异步和同步比较
![](./imgs/verl/sync_vs_async.png)

### verl 相关常见指令
```
export TORCH_CUDA_ARCH_LIST="8.6"
export VERL_DEBUG_QWEN3_VL=1
```
### verl 中有关 3D jagged 的问题（positon ids）
jagged tensor 是一种特殊的 nested tensor，形状可以是 `(bs,seq)`，但每个 batch 里的 `seq` 长度不一样，其中值的存储方式是一个扁平的 `(total_nnz,)` tensor + 一个 offsets 来记录每条序列的起止位置。
verl 里用它来存 `position_ids`，然而因为 mRoPE 的 `position_ids` 是 `(bs,4,seq)` 这种 3D case，jagged tensor 对 3D 的支持似乎存在问题，主要表现为值的存储逻辑混乱。
**定位性修改**
这些改动主要是为了确认 `position_ids` 到底在哪一层坏掉。

- [verl/experimental/agent_loop/agent_loop.py](E:/学习历程/RL_project/verl/verl/experimental/agent_loop/agent_loop.py)
  作用：确认 prompt rebuild 后，多模态 token、`image_grid_thw`、`vision_position_ids`、`final_position_ids` 在 agent loop 侧是否正常。
  解决的问题：一开始发现了“prompt rebuild 把视觉 token 丢掉”这一类前端问题，修复后确认 agent loop 产出的 `final_position_ids=(bs,4,seq)` 本身是合理的。

- [verl/models/transformers/qwen3_vl.py](E:/学习历程/RL_project/verl/verl/models/transformers/qwen3_vl.py)
  作用：在 `forward before language_model`、`rotary_emb`、`apply_rotary_pos_emb` 三层加 debug。
  解决的问题：把错误从“Qwen3-VL rotary 崩溃”收敛成“某些 rank 进入 text model 时 `position_ids` 已经坏了”，最后确认过一版坏值是 `(3,1,8)`，后面又确认修复后 `rotary/cos` 已恢复正常。

- [verl/workers/engine/fsdp/transformer_impl.py](E:/学习历程/RL_project/verl/verl/workers/engine/fsdp/transformer_impl.py)
  作用：给 engine 的 `prepare_model_inputs()` 加 `pre-rmpad/post-rmpad` 日志。
  解决的问题：确认坏值最早出现在 engine 消费 `position_ids` 的阶段，而不是 Qwen3-VL text model 自己算坏的。

**实质修复**
这些改动才是真正为了解掉 3D jagged `position_ids` 的错误。

- [verl/experimental/agent_loop/agent_loop.py](E:/学习历程/RL_project/verl/verl/experimental/agent_loop/agent_loop.py)
  修改：
  - 在`_compute_multi_modal_inputs()`中，为了拿到`multi_modal_inputs`，会将 prompt 和 rollout 得到的 response 的 token ids 拼在一起，然后 decode 成 text + vision token 的形式，最后再传入`processor`重建成 `multi_modal_inputs`。然而 decode 跳过了 special token，导致重建时的视觉相关token丢失了。现在改成直接从原 raw prompt 重建 `multi_modal_inputs`，避免 decode 这个环节。
  解决的问题：
  - 修复了第零类错误：prompt rebuild 把视觉 token 丢掉了，导致后面 `position_ids` 计算时根本没了视觉 token 的位置，最终 `position_ids` 维度和数值都错了。

- [verl/utils/tensordict_utils.py](E:/学习历程/RL_project/verl/verl/utils/tensordict_utils.py)
  修改：
  - 给 3D nested `position_ids` 显式补 `_ragged_idx = 2`
  - 在 `index_select_tensor_dict()`、`chunk_tensordict()` 这些“重建 nested tensor”的地方，把这个 metadata 补回去
  解决的问题：
  - 修复了第一类错误：把rollout数据喂给actor时，会把 batch 划分成 micro-batch ，这会重建 tensor，但重建后 `_ragged_idx` 丢失，导致 `position_ids` 被错解释成 `(4,1,4)` / `cos=(1,4,128)` 这种明显坏形状。
  - `_ragged_idx`指的是 jagged tensor 中长短不一的那个维度的索引。对于 `(bs,4,seq)` 的 3D case，`_ragged_idx=2` 就是告诉它第三维是 jagged 的，重建时才会正确处理。

- [verl/workers/engine/fsdp/transformer_impl.py](E:/学习历程/RL_project/verl/verl/workers/engine/fsdp/transformer_impl.py)
  修改：
  - 把旧的
    `position_ids.values().unsqueeze(1)`
    改成对 3D `position_ids` 显式重建 `position_ids_rmpad`
  - 先 `to_padded_tensor(...)`，再按 `input_ids.offsets().diff()` 的真实长度拼成 `(4,1,total_nnz)`
  解决的问题：
  - 修复了第二类错误：即使 `_ragged_idx` 还在，`3D jagged NestedTensor.values()` 仍可能返回错误的内部 packed layout，比如 `(4,8)` 而不是 `(4,296)`。

- [verl/workers/utils/padding.py](E:/学习历程/RL_project/verl/verl/workers/utils/padding.py)
  修改：
  - 不再把 3D 的 `position_ids` 存成 jagged nested tensor
  - 对 `(bs,4,seq)` 的 mRoPE `position_ids`，直接保留 dense padded 形式
  解决的问题：
  - 修复了更根上的问题：`torch.nested.as_nested_tensor(position_ids_list, layout=torch.jagged)` 对 `(4,seq)` 这种 3D case 本身就不稳定，后面即使 engine 不用 `.values()`，但最终访问的数值也是错的。
  - 这是最后把 `position_ids_preview=[0,1,2,3,0,0,0,0]` 这种坏值清掉的关键一步。

- [verl/workers/engine/fsdp/transformer_impl.py](E:/学习历程/RL_project/verl/verl/workers/engine/fsdp/transformer_impl.py)
  额外配套修改：
  - 新增了 dense 3D `position_ids` 的 remove-padding 切片逻辑
  - 用 `attention_mask` 从 `(bs,4,seq)` 显式切成 `(4,1,total_nnz)`
  解决的问题：
  - 让上面 `padding.py` 的新数据形态能在 engine 里继续工作，不再依赖 nested 3D `position_ids`。
