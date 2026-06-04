#!/bin/bash

#SBATCH -A m5031_g
#SBATCH -J stft-ddp
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:05:00
#SBATCH -N 1
#SBATCH --signal=USR1@120
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --image=nersc/pytorch:25.06.01
#SBATCH --module=gpu,nccl-plugin
#SBATCH -L scratch
#SBATCH -o stft-ddp-%j.out
#SBATCH -e stft-ddp-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adamrupe@lbl.gov

export OMP_NUM_THREADS=8 
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-$((10000 + SLURM_JOB_ID % 50000))}"

export WANDB_MODE="disabled"

srun --cpu-bind=cores shifter bash -c \
    'unset NCCL_CROSS_NIC; exec torchrun "$@"' bash \
    --nnodes="${SLURM_NNODES}" \
    --nproc-per-node=4 \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv-id="${SLURM_JOB_ID}" \
    ../train.py
