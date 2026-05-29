#!/bin/bash

# Multi-node DDP launch for StFT on NERSC-style Slurm systems.
# torchrun reference: https://docs.pytorch.org/docs/stable/elastic/run.html
# Slurm provides job/node metadata such as SLURM_JOB_ID, SLURM_NNODES, and
# SLURM_JOB_NODELIST to batch scripts.

#SBATCH -A mXXXX_g
#SBATCH -J stft-ddp
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH -N 2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH -c 32
#SBATCH --image=nersc/pytorch:25.06.01
#SBATCH --module=gpu,nccl-plugin
#SBATCH -L scratch
#SBATCH -o stft-ddp-%j.out
#SBATCH -e stft-ddp-%j.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
MASTER_PORT="${MASTER_PORT:-$((10000 + SLURM_JOB_ID % 50000))}"

export MASTER_ADDR
export MASTER_PORT
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "SLURM_NNODES=${SLURM_NNODES}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"

srun --cpu-bind=cores shifter torchrun \
    --nnodes="${SLURM_NNODES}" \
    --nproc-per-node="${GPUS_PER_NODE}" \
    --rdzv-backend=c10d \
    --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
    --rdzv-id="${SLURM_JOB_ID}" \
    train.py
