from __future__ import annotations

import json
import os
import re
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from stft.distributed import launch_contract


SLURM_ENV_KEYS = {
    "job_id": "SLURM_JOB_ID",
    "job_name": "SLURM_JOB_NAME",
    "nodelist": "SLURM_JOB_NODELIST",
    "nnodes": "SLURM_NNODES",
    "job_num_nodes": "SLURM_JOB_NUM_NODES",
    "ntasks": "SLURM_NTASKS",
    "ntasks_per_node": "SLURM_NTASKS_PER_NODE",
    "gpus": "SLURM_GPUS",
    "gpus_per_node": "SLURM_GPUS_PER_NODE",
    "gpus_on_node": "SLURM_GPUS_ON_NODE",
    "cpus_per_task": "SLURM_CPUS_PER_TASK",
    "partition": "SLURM_JOB_PARTITION",
    "qos": "SLURM_QOS",
    "account": "SLURM_JOB_ACCOUNT",
    "procid": "SLURM_PROCID",
    "localid": "SLURM_LOCALID",
    "submit_dir": "SLURM_SUBMIT_DIR",
    "cluster_name": "SLURM_CLUSTER_NAME",
}


def collect_launch_metadata(
    *,
    device: torch.device | None = None,
    local_rank: int | None = None,
    rank: int | None = None,
    world_size: int | None = None,
    start_epoch: int | None = None,
    resume_from_checkpoint: bool | None = None,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    timestamp = timestamp or datetime.now(timezone.utc)
    slurm = _collect_env(SLURM_ENV_KEYS)
    distributed_initialized = dist.is_available() and dist.is_initialized()

    if distributed_initialized:
        backend = str(dist.get_backend())
        observed_rank = dist.get_rank()
        observed_world_size = dist.get_world_size()
    else:
        backend = None
        observed_rank = rank
        observed_world_size = world_size

    return {
        "timestamp": _format_timestamp(timestamp),
        "slurm": slurm,
        "torch_distributed": {
            "launch_contract": launch_contract(),
            "available": dist.is_available(),
            "initialized": distributed_initialized,
            "backend": backend,
            "world_size": observed_world_size,
            "rank": observed_rank,
            "local_rank": local_rank,
            "env_rank": os.environ.get("RANK"),
            "env_world_size": os.environ.get("WORLD_SIZE"),
            "master_addr": os.environ.get("MASTER_ADDR"),
            "master_port": os.environ.get("MASTER_PORT"),
        },
        "runtime": {
            "hostname": socket.gethostname(),
            "device": str(device) if device is not None else None,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_current_device": (
                torch.cuda.current_device() if torch.cuda.is_available() else None
            ),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        "training": {
            "resume_from_checkpoint": resume_from_checkpoint,
            "start_epoch": start_epoch,
        },
    }


def write_launch_metadata(
    save_path: str | Path,
    *,
    device: torch.device | None = None,
    local_rank: int | None = None,
    rank: int | None = None,
    world_size: int | None = None,
    start_epoch: int | None = None,
    resume_from_checkpoint: bool | None = None,
    timestamp: datetime | None = None,
) -> tuple[Path, dict[str, Any]]:
    timestamp = timestamp or datetime.now(timezone.utc)
    metadata = collect_launch_metadata(
        device=device,
        local_rank=local_rank,
        rank=rank,
        world_size=world_size,
        start_epoch=start_epoch,
        resume_from_checkpoint=resume_from_checkpoint,
        timestamp=timestamp,
    )
    launch_id = _make_launch_id(timestamp, metadata["slurm"]["job_id"])

    launch_dir = _unique_launch_dir(Path(save_path) / "launches", launch_id)
    metadata["launch_id"] = launch_dir.name
    launch_path = launch_dir / "launch.json"
    tmp_path = launch_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    tmp_path.replace(launch_path)
    return launch_path, metadata


def _collect_env(keys: dict[str, str]) -> dict[str, str | None]:
    return {name: os.environ.get(env_name) for name, env_name in keys.items()}


def _make_launch_id(timestamp: datetime, job_id: str | None) -> str:
    time_part = timestamp.astimezone(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S-%fZ")
    job_part = f"job{job_id}" if job_id else "local"
    return f"{time_part}_{_sanitize_path_component(job_part)}"


def _unique_launch_dir(root: Path, launch_id: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    candidate = root / launch_id
    suffix = 2
    while True:
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            candidate = root / f"{launch_id}-{suffix}"
            suffix += 1


def _sanitize_path_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return sanitized or "unknown"


def _format_timestamp(timestamp: datetime) -> str:
    return timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
