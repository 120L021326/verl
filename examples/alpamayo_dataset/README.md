# Alpamayo Dataset

`alpamayo_dataset` 是一个用于从本地 Physical AI AV 数据集中构造 Alpamayo 多模态训练样本的独立示例目录。它的核心流程是：

1. 从本地 PAI 数据目录读取 `clip_index.parquet`。
2. 生成轻量级 `train.parquet` / `val.parquet` 元数据。
3. 在 `AlpamayoDemoDataset` 运行时按 `clip_id` 和 `t0_us` 加载图像帧、历史轨迹和未来轨迹。
4. 构造成 verl 数据管线可消费的 sample 字典。

## 1. 项目目录和文件介绍

```text
examples/alpamayo_dataset/
├── README.md
├── alpamayo_dataset.py
├── requirements.txt
├── clip_metadata_parquet/
│   ├── train.parquet
│   └── val.parquet
├── scripts/
│   └── build_metadata.sh
├── tools/
│   └── load_dataset_pre.py
└── utils/
    ├── delta_tokenizer.py
    ├── load_physical_aiavdataset.py
    └── pai_utils.py
```

### `alpamayo_dataset.py`

主数据集文件，定义 `AlpamayoDemoDataset`。

主要职责：

- 读取 `.jsonl`、`.json` 或 `.parquet` 元数据文件。
- 根据每条 metadata 中的 `clip_id` 和 `t0_us` 加载本地 PAI 数据。
- 读取 4 个 camera 的图像帧。
- 读取 egomotion，并构造历史轨迹和未来轨迹。
- 将历史轨迹编码成 Alpamayo trajectory tokens。
- 构造包含图像和文本 prompt 的 `raw_prompt`。
- 返回 verl 数据管线可用的 sample 字典。

文件底部的 `if __name__ == "__main__"` 是一个最小验证入口，会读取 `clip_metadata_parquet/val.parquet` 并打印前若干条 sample。

### `requirements.txt`

列出本目录额外依赖：

```text
physical_ai_av
qwen_vl_utils
```

可能还需要其他依赖，请补充。

### `clip_metadata_parquet/`

存放生成后的轻量 metadata parquet 文件：

- `train.parquet`
- `val.parquet`

这些文件不是原始 PAI 数据，只保存训练/验证索引用的信息，例如：

- `clip_id`
- `t0_us`
- `idx`
- `data_source`

真正的视频、egomotion、feature metadata 仍然来自本地 PAI 数据目录。

### `scripts/build_metadata.sh`

自包含的 metadata 构建脚本。

它会调用 `tools/load_dataset_pre.py`，从本地 PAI 数据目录生成：

- `clip_metadata_parquet/train.parquet`
- `clip_metadata_parquet/val.parquet`

常用参数：

- `--data-dir`: 本地 PAI 数据根目录。
- `--metadata-output-dir`: 输出 parquet 的目录，默认是 `clip_metadata_parquet`。
- `--chunk-ids`: 指定 chunk，例如 `3116` 或 `3116-3120`。
- `--num-samples`: 在过滤后的候选数据中随机抽样指定条数。
- `--random-seed`: 抽样随机种子，默认 `11`。
- `--clip-ids-file`: 从 JSON 文件指定 clip 列表。
- `--val-ratio`: 验证集比例，默认 `0.1`。
- `--force-rebuild`: 已存在 parquet 时强制重建。

### `tools/load_dataset_pre.py`

metadata 生成脚本。

主要逻辑：

- 读取本地 PAI 数据目录下的 `clip_index.parquet`。
- 可选按 `chunk` 过滤。
- 可选按 `--clip-ids-file` 指定 clip。
- 可选用 `--num-samples` 做无放回均匀随机抽样。
- 为每条记录写入 `idx` 和 `data_source`。
- 按 `--val-ratio` 切分 train / val。
- 写出 `train.parquet` 和 `val.parquet`。

注意：`build_metadata.sh` 默认传入 `--skip-download`，所以不会下载 Hugging Face 数据，只使用已有的本地 PAI 数据。

### `utils/pai_utils.py`

本地 PAI 数据访问接口。

主要职责：

- 读取 PAI 数据目录下的 `features.csv`。
- 读取 `clip_index.parquet`。
- 读取 `metadata/feature_presence.parquet`。
- 根据 `clip_id` 和 feature 名称定位具体 chunk 文件。
- 支持读取 parquet feature、zip 中的 egomotion 和 camera video。

### `utils/load_physical_aiavdataset.py`

把 `PhysicalAIAVDatasetLocalInterface` 读出的原始数据转换成 Alpamayo 所需格式。

主要输出：

- camera image frames
- camera indices
- ego history trajectory
- ego future trajectory
- relative / absolute timestamps
- `clip_id`
- `t0_us`

### `utils/delta_tokenizer.py`

轨迹 tokenizer。

用于把历史轨迹编码成 Alpamayo prompt 中的离散 trajectory token，例如 `<i3001>` 这类 token id 形式。

## 2. Dataset 输出字段说明

`AlpamayoDemoDataset.__getitem__()` 返回一个 `dict[str, Any]`。返回内容包括原始 metadata 字段，以及运行时补充的字段。

### metadata 原始字段

从 `train.parquet` / `val.parquet` 读取，通常包括：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `clip_id` | `str` | PAI 数据中的 clip id。 |
| `t0_us` | `int` | 当前样本的关键时间戳，单位是 microseconds。 |
| `idx` | `int` | metadata 生成时写入的样本序号。 |
| `data_source` | `str` | 数据来源标识，默认是 `alpamayo_physical_ai_av`。 |

### dataset 运行时新增字段

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `raw_prompt` | `list[dict]` | 多模态对话 prompt，包含 system / user / assistant 三段。user 中包含图像帧和历史轨迹 token 占位替换后的文本。 |
| `dummy_tensor` | `torch.Tensor` | verl 数据管线需要的占位 tensor，当前为 `torch.tensor([0], dtype=torch.uint8)`。 |
| `ground_truth` | `Any` | 监督答案字段。优先从原始 row 的 `ground_truth`、`reward_model.ground_truth`、`answer`、`coc`、`target` 中解析；没有则为空字符串。 |
| `reward_model` | `dict` | reward 相关字段，至少包含 `ground_truth`。 |
| `extra_info` | `dict` | 附加信息，包含 index、clip 信息和轨迹 tensor。 |
| `index` | `int` | 样本 index，来自 `extra_info["index"]`。 |
| `tools_kwargs` | `dict` | 工具调用参数，默认 `{}`。 |
| `interaction_kwargs` | `dict` | 交互参数，默认 `{}`。 |

### `raw_prompt` 结构

`raw_prompt` 是一个三轮 message 列表：

```python
[
    {"role": "system", "content": [{"type": "text", "text": "..."}]},
    {
        "role": "user",
        "content": [
            {"type": "image", "image": <PIL.Image.Image>},
            ...,
            {"type": "text", "text": "<|traj_history_start|>...<|traj_history_end|>output ..."},
        ],
    },
    {"role": "assistant", "content": [{"type": "text", "text": "<|cot_start|>"}]},
]
```

其中 image 来自 4 个 camera、每个 camera 多帧图像；text 中的历史轨迹占位符会被真实 trajectory token 替换。

### `extra_info` 主要内容

| 字段 | 类型 / shape | 说明 |
| --- | --- | --- |
| `index` | `int` | 样本 index。 |
| `clip_id` | `str` | 当前样本 clip id。 |
| `t0_us` | `int` | 当前采样时间戳。 |
| `hist_token_count` | `int` | 历史轨迹 token 数。 |
| `ego_history_xyz` | `torch.Tensor`, shape `[1, 1, 16, 3]` | 历史 ego 位置，已经转换到 t0 局部坐标系。 |
| `ego_history_rot` | `torch.Tensor`, shape `[1, 1, 16, 3, 3]` | 历史 ego 旋转矩阵。 |
| `ego_future_xyz` | `torch.Tensor`, shape `[1, 1, 64, 3]` | 未来 ego 位置，已经转换到 t0 局部坐标系。 |
| `ego_future_rot` | `torch.Tensor`, shape `[1, 1, 64, 3, 3]` | 未来 ego 旋转矩阵。 |

默认轨迹设置来自 `load_physical_aiavdataset.py`：

- history: 16 steps
- future: 64 steps
- time step: 0.1s
- image frames: 4 frames per camera
- camera: front wide、cross left、cross right、front tele

这里有两个不同的历史窗口，容易混淆：

- **trajectory history window**: 用于 `ego_history_xyz` / `ego_history_rot`。
  默认 `num_history_steps=16`、`time_step=0.1s`，因此会采样 16 个历史轨迹点：`t0-1.5s, t0-1.4s, ..., t0-0.1s, t0`。代码中还会检查`t0_us > num_history_steps * time_step * 1_000_000`，所以默认要求`t0_us > 1.6s`，否则历史轨迹窗口不足。
- **image frame window**: 用于 `image_frames`。默认 `num_frames=4`、`time_step=0.1s`，因此每个 camera 取 4 个图像时间点：`t0-0.3s, t0-0.2s, t0-0.1s, t0`。这 4 帧覆盖的是 0.3 秒时间跨度，不是 1.6 秒。

也就是说，虽然每个 camera 默认只取 4 帧图像，但 dataset 同时还会加载1.6 秒左右的 ego trajectory history，并把这段历史轨迹编码进 prompt 中的trajectory tokens。因此用于生成 metadata 的 `t0_us` 需要优先满足轨迹历史窗口，而不仅仅满足图像取帧窗口。

## 3. 使用步骤

以下步骤假设你已经有本地 PAI 数据目录，例如：

```text
/share/datasets/Alpamayo_pai_av_big
```

该目录至少需要包含：

```text
features.csv
clip_index.parquet
metadata/
camera/
labels/
```

其中本数据集默认会使用 4 个 camera feature 和 `labels/egomotion`。

### Step 1: 进入目录

```bash
cd examples/alpamayo_dataset
```

### Step 2: 安装依赖

如果你在 verl 项目的 Python 环境中运行，先安装本目录额外依赖：

```bash
pip install -r requirements.txt
```

如果当前环境还缺少通用运行依赖，可以安装：

```bash
pip install pandas pyarrow numpy torch pillow transformers einops scipy
```

`build_metadata.sh` 是 bash 脚本。在 Windows 上建议使用 WSL、Git Bash，或其他可运行 bash 的环境。

### Step 3: 生成 parquet metadata

生成指定 chunk 的全部 metadata：

```bash
./scripts/build_metadata.sh \
  --data-dir /share/datasets/Alpamayo_pai_av_big \
  --chunk-ids 3116 \
  --force-rebuild
```

从多个 chunk 中随机抽样固定条数：

```bash
./scripts/build_metadata.sh \
  --data-dir /share/datasets/Alpamayo_pai_av_big \
  --chunk-ids 3116-3120 \
  --num-samples 320 \
  --random-seed 42 \
  --force-rebuild
```

这里的含义是：

- 先从 chunk `3116, 3117, 3118, 3119` 里筛选候选 clip。
- 再无放回均匀随机抽样 1000 条。
- 再按 `--val-ratio` 切分 train / val。

默认输出：

```text
clip_metadata_parquet/train.parquet
clip_metadata_parquet/val.parquet
```

如果 metadata 已经存在，脚本会默认退出，不覆盖旧文件。需要重建时加：

```bash
--force-rebuild
```

### Step 4: 检查生成结果

可以用 Python 快速查看 parquet 行数：

```bash
python - <<'PY'
import pandas as pd

train = pd.read_parquet("clip_metadata_parquet/train.parquet")
val = pd.read_parquet("clip_metadata_parquet/val.parquet")

print("train rows:", len(train))
print("val rows:", len(val))
print(train.head())
PY
```
```bash
python - <<'PY'
import pandas as pd

train = pd.read_parquet("/workspace/verl/examples/alpamayo_demo3/data/train.parquet")

print("train rows:", len(train))
print(train.head())
PY
```
### Step 5: 运行 dataset 验证

`alpamayo_dataset.py` 底部提供了一个简单验证入口：

```bash
python alpamayo_dataset.py
```

运行前需要确认两个路径：

1. tokenizer 路径存在：

```python
AutoTokenizer.from_pretrained("/workspace/Alpamayo-R1-10B-vlm/")
```

2. PAI 数据路径正确：

```python
DEFAULT_DATA_LOCAL_DIR = "/share/datasets/Alpamayo_pai_av_big/"
```

如果你的路径不同，可以临时修改 `alpamayo_dataset.py` 中的这两个路径，或在自己的脚本中这样实例化：

```python
from transformers import AutoTokenizer
from alpamayo_dataset import AlpamayoDemoDataset

tokenizer = AutoTokenizer.from_pretrained("/path/to/Alpamayo-R1-10B-vlm")

dataset = AlpamayoDemoDataset(
    data_files="clip_metadata_parquet/val.parquet",
    tokenizer=tokenizer,
    config={"data_local_dir": "/path/to/Alpamayo_pai_av_big"},
    max_samples=10,
)

sample = dataset[0]
print(sample.keys())
print(sample["raw_prompt"])
print(sample["extra_info"].keys())
```

如果成功，说明：

- metadata parquet 可以正常读取。
- `clip_id` 能在本地 PAI 数据中解析到对应 chunk。
- camera 图像可以读取并转为 PIL image。
- egomotion 可以读取并构造历史/未来轨迹。
- `raw_prompt` 和 `extra_info` 能被正确生成。
