#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --output=output-%N.txt
#SBATCH --partition=gpu-1xA100,gpu-2xA100,gpu-8xA100
source .bashrc
cd joklar/parallel
hostname
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo Number of GPUS: $NUM_GPUS
for i in $(seq 0 $((NUM_GPUS - 1))); do
  export CUDA_VISIBLE_DEVICES=$i
  client.py &
done
