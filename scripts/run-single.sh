#!/bin/bash

#SBATCH -A m5031_g
#SBATCH -J pytorch-gpu
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH -t 12:00:00
#SBATCH -n 1
#SBATCH --gpus-per-task=1
#SBATCH --image=nersc/pytorch:25.06.01
#SBATCH --module=gpu,nccl-plugin

#SBATCH -o ./slurm/slurm-%j.out
#SBATCH -e ./slurm/slurm-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adamrupe@lbl.gov

set -euo pipefail

srun --cpu-bind=cores shifter python train.py