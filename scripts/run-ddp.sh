#!/bin/bash

#SBATCH -A m5031_g
#SBATCH -J stft-ddp
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:05:00
#SBATCH -N 2
#SBATCH --signal=USR1@120
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --image=nersc/pytorch:25.06.01
#SBATCH --module=gpu,nccl-plugin
#SBATCH -L scratch
#SBATCH -o ../slurm/stft-ddp-%j.out
#SBATCH -e ../slurm/stft-ddp-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adamrupe@lbl.gov

export OMP_NUM_THREADS=8 
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-$((10000 + SLURM_JOB_ID % 50000))}"

export WANDB_MODE="disabled"

srun --cpu-bind=cores shifter bash -c \
    'unset NCCL_CROSS_NIC; exec python "$@"' bash \
    ../train.py
