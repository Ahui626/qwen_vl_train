#!/bin/bash
# 多卡训练启动脚本
# 使用方法: 
#   bash train_ddp.sh                              # 从头开始训练
#   bash train_ddp.sh output/checkpoint-1500       # 从指定checkpoint继续训练

# 设置GPU（从2号开始，使用4张4090）
export CUDA_VISIBLE_DEVICES=2,3,4,5

# 优化显存分配
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 获取checkpoint路径参数（可选）
CHECKPOINT=${1:-""}

# 使用torchrun启动分布式训练
# nproc_per_node=4 表示使用4个GPU
if [ -z "$CHECKPOINT" ]; then
    echo "=================================================="
    echo "从头开始训练..."
    echo "=================================================="
    torchrun --nproc_per_node=4 \
        --master_port=29500 \
        train.py --config config.yaml
else
    echo "=================================================="
    echo "从checkpoint继续训练: $CHECKPOINT"
    echo "=================================================="
    torchrun --nproc_per_node=4 \
        --master_port=29500 \
        train.py --config config.yaml \
        --resume_from_checkpoint "$CHECKPOINT"
fi

