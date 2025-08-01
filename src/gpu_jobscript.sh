#!/bin/sh
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=64G           
#SBATCH --time=6:00:00
#SBATCH --job-name=packets
#SBATCH --output=../slurm_outputs/output_%A_%a.txt
#SBATCH --error=../slurm_outputs/error_%A_%a.txt

module load slurm
module load python/3.8
source ../vna/bin/activate
export MASTER_PORT=$((12355 + SLURM_ARRAY_TASK_ID))

# Run the Python script with the system_size argument
torchrun --nproc_per_node=2 main.py 