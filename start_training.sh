#!/bin/bash
# Set environment variables for optimized CPU training
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=""
export PYTHONPATH=/home/ubuntu/Generative-Flex

# Start training with proper logging
python train_mmmu_cpu.py 2>&1 | tee logs/training_$(date +%Y%m%d_%H%M%S).log
