#!/bin/bash
# MacBook 依赖安装脚本

set -e

echo "=== 检查 Python 版本 ==="
python --version
which python

echo ""
echo "=== 步骤 1: 安装 PyTorch (MacBook CPU/MPS 版本) ==="
# 先安装 PyTorch（MacBook 版本，支持 MPS）
python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cpu

echo ""
echo "=== 步骤 2: 安装其他依赖 ==="
python -m pip install -r requirements_macbook.txt

echo ""
echo "=== 步骤 3: 安装可选依赖（如果需要） ==="
# human_eval 是可选的，用于代码评估
python -m pip install human_eval || echo "human_eval 安装失败（可选）"

echo ""
echo "=== 安装完成！ ==="
echo "注意：flash-attn 和 deepspeed 已跳过（MacBook 不需要 CUDA）"

