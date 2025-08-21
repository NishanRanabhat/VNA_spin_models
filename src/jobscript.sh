#!/bin/sh
#SBATCH --partition=h200q
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=64G           # Request 128GB of memory total
#SBATCH --time=2:00:00
#SBATCH --job-name=packets
#SBATCH --array=0-6         # Array job with 4 tasks (for system_size values 10, 20, 30, 40)
#SBATCH --output=../slurm_outputs/output_%A_%a.txt
#SBATCH --error=../slurm_outputs/error_%A_%a.txt

module load slurm
module load python/3.8
source ../vna/bin/activate
export MASTER_PORT=$((29500 + 100 + $SLURM_ARRAY_TASK_ID * 100))

# Define the system_size values
annealing_time=(32000 20000 16000 9000 7000 5000 3000)

# Get the system_size for this task
annealing_time=${annealing_time[$SLURM_ARRAY_TASK_ID]}

# Run the Python script with the system_size argument
python main.py $annealing_time