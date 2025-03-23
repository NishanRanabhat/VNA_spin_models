#!/bin/sh
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=128G           # Request 128GB of memory total
#SBATCH --time=0:45:00
#SBATCH --job-name=packets
#SBATCH --array=0-8          # Array job with 4 tasks (for system_size values 10, 20, 30, 40)
#SBATCH --output=./slurm_outputs/output_%A_%a.txt
#SBATCH --error=./slurm_outputs/error_%A_%a.txt

module load slurm
module load python/3.8
source ../vnaenv/bin/activate
export MASTER_PORT=$((12355 + SLURM_ARRAY_TASK_ID))

# Define the system_size values
Tf=(0.0 0.5 0.6 0.7 0.8 1.0 1.5 2.5 3.5)

# Get the system_size for this task
Tf=${Tf[$SLURM_ARRAY_TASK_ID]}

# Run the Python script with the system_size argument
python main.py $Tf