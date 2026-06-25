#!/bin/bash

#SBATCH -A m5031_g
#SBATCH -J stft-4
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -N 4

#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --image=nersc/pytorch:25.06.01
#SBATCH --module=gpu,nccl-plugin
#SBATCH -o /pscratch/sd/m/mcho4/StFT-data-parellel-new/slurm/stft-ddp-%j.out
#SBATCH -e /pscratch/sd/m/mcho4/StFT-data-parellel-new/slurm/stft-ddp-%j.err
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=

export OMP_NUM_THREADS=8 
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-$((10000 + SLURM_JOB_ID % 50000))}"

export WANDB_MODE="disabled"

srun --cpu-bind=cores shifter bash -c \
    'unset NCCL_CROSS_NIC; exec python "$@"' bash \
    /pscratch/sd/m/mcho4/StFT-data-parellel-new/train.py --config-name scaling \
    save_path=/pscratch/sd/m/mcho4/StFT-data-parellel-new/experiments/debug