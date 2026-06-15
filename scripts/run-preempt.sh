#!/bin/bash

#SBATCH -A m5031_g
#SBATCH -J stft-nocat
#SBATCH -C gpu
#SBATCH -q preempt
#SBATCH -t 24:00:00
#SBATCH -N 4
#SBATCH --signal=USR1@360
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --image=nersc/pytorch:25.06.01
#SBATCH --module=gpu,nccl-plugin
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH -o /pscratch/sd/a/atrupe/StFT/slurm/stft-ddp-%j.out
#SBATCH -e /pscratch/sd/a/atrupe/StFT/slurm/stft-ddp-%j.err
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=adamrupe@lbl.gov

export OMP_NUM_THREADS=8 
export MASTER_ADDR="$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n 1)"
export MASTER_PORT="${MASTER_PORT:-$((10000 + SLURM_JOB_ID % 50000))}"

# export WANDB_MODE="disabled"

srun --cpu-bind=cores shifter bash -c \
    'unset NCCL_CROSS_NIC; exec python "$@"' bash \
    /pscratch/sd/a/atrupe/StFT/train.py

status=$?
if [ "$status" -ne 0 ]; then
    exit "$status"
fi

sleep 120 # make sure a process is still running for slurm to send SIGKILL to