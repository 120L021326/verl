### before run sh
```bash
cd verl
export TORCH_CUDA_ARCH_LIST="8.6"
export PYTHONPATH="$PWD/alpamayo/src:$PWD/alpamayo/finetune:$PWD/alpamayo/finetune/rl/models"
export TENSORBOARD_DIR=/workspace/tensorboard_log/alpamayo_demo
```