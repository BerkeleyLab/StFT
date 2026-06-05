#!/bin/bash

#SBATCH -A m5031_g
#SBATCH -J stft-ddp
#SBATCH -C gpu
#SBATCH -q debug_preempt
#SBATCH -t 00:20:00
#SBATCH -N 2
#SBATCH --signal=USR1@120
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --image=nersc/pytorch:25.06.01
#SBATCH --module=gpu,nccl-plugin
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o ../slurm/stft-ddp-%j.out
#SBATCH -e ../slurm/stft-ddp-%j.err
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=adamrupe@lbl.gov

export OMP_NUM_THREADS=8 
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-$((10000 + SLURM_JOB_ID % 50000))}"

export WANDB_MODE="disabled"

srun --cpu-bind=cores shifter bash -c \
    'unset NCCL_CROSS_NIC; exec python "$@"' bash \
    ../train.py

sleep 120 # make sure a process is still running for slurm to send SIGKILL to