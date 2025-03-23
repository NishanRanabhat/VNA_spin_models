module load slurm
srun -p gpudebugq -t 1:00:00 --pty --gpus=1 bash