#!/bin/bash

#SBATCH -A m5031_g
#SBATCH -J stft-preempt
#SBATCH -C gpu
#SBATCH -q preempt
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --gpus-per-task=1
#SBATCH --image=nersc/pytorch:25.06.01
#SBATCH --module=gpu,nccl-plugin
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -L scratch

#SBATCH -o ./slurm/slurm-%j.out
#SBATCH -e ./slurm/slurm-%j.err
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=adamrupe@lbl.gov

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

srun --cpu-bind=cores shifter python train.py

# Helps Slurm record PREEMPTED cleanly if the srun-launched process exits
# after handling the preemption SIGTERM.
sleep 120
