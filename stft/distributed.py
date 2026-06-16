import os
import socket

import torch
import torch.distributed as dist


DDP_LAUNCHER = None


def using_torchrun_environment() -> bool:
    return all(name in os.environ for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))


def using_slurm_direct_environment() -> bool:
    return "SLURM_PROCID" in os.environ and int(os.environ.get("SLURM_NTASKS", "1")) > 1


def launch_contract() -> str:
    if DDP_LAUNCHER is not None:
        return DDP_LAUNCHER
    if using_torchrun_environment():
        return "torchrun"
    if using_slurm_direct_environment():
        return "slurm-direct"
    return "plain-python"


def normalize_slurm_environment() -> None:
    global DDP_LAUNCHER
    if DDP_LAUNCHER is not None:
        return

    if using_torchrun_environment():
        DDP_LAUNCHER = "torchrun"
        return

    if using_slurm_direct_environment():
        DDP_LAUNCHER = "slurm-direct"
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        if torch.cuda.is_available():
            visible_count = torch.cuda.device_count()
            if visible_count <= 0:
                raise RuntimeError("CUDA is available but no CUDA devices are visible")
            slurm_localid = int(os.environ.get("SLURM_LOCALID", rank))
            os.environ["LOCAL_RANK"] = str(slurm_localid % visible_count)
        else:
            os.environ["LOCAL_RANK"] = os.environ.get("SLURM_LOCALID", str(rank))


def distributed_launch_detected() -> bool:
    return launch_contract() in {"torchrun", "slurm-direct"}


def distributed_is_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def distributed_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def setup_distributed() -> tuple[torch.device, int, int, int]:
    normalize_slurm_environment()
    if distributed_is_enabled(): # handle if this function has already been called
        local_rank = distributed_local_rank()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            device = torch.device("cpu")
        return device, local_rank, rank, world_size

    if distributed_launch_detected():
        missing = [
            name
            for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
            if name not in os.environ
        ]
        if missing:
            raise RuntimeError(f"Distributed launch is missing required environment variables: {missing}")
        local_rank = int(os.environ["LOCAL_RANK"])
        if torch.cuda.is_available():
            visible_count = torch.cuda.device_count()
            if visible_count <= 0:
                raise RuntimeError("CUDA is available but no CUDA devices are visible")
            if local_rank < 0 or local_rank >= visible_count:
                raise RuntimeError(
                    "LOCAL_RANK is not a valid CUDA-visible device index: "
                    f"local_rank={local_rank}, cuda_device_count={visible_count}, "
                    f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}."
                )
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"
        init_kwargs = {"backend": backend}
        if device.type == "cuda":
            init_kwargs["device_id"] = device
        dist.init_process_group(**init_kwargs)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            device = torch.device("cpu")

    return device, local_rank, rank, world_size


def cleanup_distributed() -> None:
    if distributed_is_enabled():
        dist.destroy_process_group()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def reduce_sum(value: torch.Tensor, distributed: bool = False) -> torch.Tensor:
    value = value.detach().clone()
    if distributed:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def reduce_max(value: torch.Tensor, distributed: bool = False) -> torch.Tensor:
    value = value.detach().clone()
    if distributed:
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
    return value


def reduce_mean_from_counts(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    distributed: bool = False,
) -> torch.Tensor:
    total = reduce_sum(torch.stack([numerator.detach(), denominator.detach()]), distributed)
    return total[0] / total[1].clamp_min(1)


def barrier(distributed: bool = False) -> None:
    if distributed:
        dist.barrier()


def log_distributed_preflight(
    device: torch.device,
    local_rank: int,
    rank: int,
    world_size: int,
) -> None:
    print(
        " ".join(
            [
                f"launch_contract={launch_contract()}",
                f"host={socket.gethostname()}",
                f"rank={rank}",
                f"local_rank={local_rank}",
                f"world_size={world_size}",
                f"device={device}",
                f"device_index={device.index if device.type == 'cuda' else '<none>'}",
                f"slurm_localid={os.environ.get('SLURM_LOCALID', '<unset>')}",
                f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}",
                f"cuda_device_count={torch.cuda.device_count()}",
            ]
        ),
        flush=True,
    )
