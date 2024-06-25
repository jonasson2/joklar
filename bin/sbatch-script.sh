#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --output=output-%N.txt
#SBATCH --partition=gpu-1xA100,gpu-2xA100,gpu-8xA100

# Load environment variables and move to the correct directory
source .bashrc
cd joklar/parallel

# Print the hostname and store it in a variable
hostname=$(hostname)
echo "Running on node: $hostname"

# Get the number of GPUs available
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo Number of GPUs: $NUM_GPUS

# Function to monitor GPU usage
monitor_gpu() {
  log_file="gpu_usage_${hostname}.log"
  if [ ! -f "$log_file" ]; then
    # Print the header if the log file does not exist
    echo "timestamp,name,index,utilization.gpu [%],utilization.memory [%],memory.total [MiB],memory.used [MiB]" > "$log_file"
  fi
  while true; do
    nvidia-smi --query-gpu=timestamp,name,index,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv,noheader,nounits >> "$log_file"
    sleep 10
  done
}

# Start monitoring in the background
monitor_gpu &
GPU_MONITOR_PID=$!

# Run client.py on each GPU in parallel
for i in $(seq 0 $((NUM_GPUS - 1))); do
  export CUDA_VISIBLE_DEVICES=$i
  client.py &
done

# Wait for all background jobs to complete
wait

# Kill the monitoring process
kill $GPU_MONITOR_PID
