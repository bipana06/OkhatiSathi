#!/bin/bash
#SBATCH --job-name=sweep_ft
#SBATCH --array=0-99 # Adjust the upper bound as needed (e.g., 0-199)
#SBATCH -c 10
#SBATCH -t 48:00:00
#SBATCH -p nvidia
#SBATCH --gres=gpu:1
#SBATCH --output=logs/sweep_run_%a.log
#SBATCH --error=logs/error_sweep_run_%a.log
#Other SBATCH commands go here

# Create the log directory if it doesn't exist
mkdir -p logs

echo "--- Environment Info (Sweep Run: $SLURM_ARRAY_TASK_ID) ---"
echo "Hostname: $(hostname)"
echo "GPU Info:"
nvidia-smi
echo "CUDA Toolkit (nvcc):"
nvcc --version || echo "nvcc not found"
echo "Loaded Modules:"
module list
echo "Python Path: $(which python)"
echo "PyTorch Info:"
# Replace 'python' if you use 'python3' or a specific path
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version built with: {torch.version.cuda}'); print(f'Device Count: {torch.cuda.device_count()}'); print(f'Current Device: {torch.cuda.current_device()}'); print(f'Device Name: {torch.cuda.get_device_name(0)}'); print(f'Device Capability: {torch.cuda.get_device_capability(0)}'); print(f'Arch List: {torch.cuda.get_arch_list()}')" || echo "PyTorch check failed"
echo "-------------------------"
#Activating conda
module purge
conda init
conda activate t2env

export CUDA_LAUNCH_BLOCKING=1
echo "Starting W&B agent (Sweep Run: $SLURM_ARRAY_TASK_ID)..."
wandb agent --count 1 md5121-new-york-university/tacotron2/w4oyzcwb # Replace with your actual sweep ID

echo "W&B Agent (Sweep Run: $SLURM_ARRAY_TASK_ID) finished."
echo "Sweep Run $SLURM_ARRAY_TASK_ID completed at: $(date)"