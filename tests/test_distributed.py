import os

import pytest

from stft import distributed as dist_utils


_DIST_ENV_VARS = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "SLURM_PROCID",
    "SLURM_NTASKS",
    "SLURM_LOCALID",
)


@pytest.fixture(autouse=True)
def clean_distributed_env(monkeypatch):
    for name in _DIST_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(dist_utils, "DDP_LAUNCHER", None)


def test_plain_python_launch_contract():
    assert dist_utils.launch_contract() == "plain-python"
    assert not dist_utils.distributed_launch_detected()
    assert dist_utils.distributed_local_rank() == 0


def test_torchrun_launch_detected_with_single_process(monkeypatch):
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")

    assert dist_utils.distributed_launch_detected()
    assert dist_utils.launch_contract() == "torchrun"
    assert dist_utils.distributed_local_rank() == 0


def test_slurm_direct_environment_normalizes_rank_vars(monkeypatch):
    monkeypatch.setenv("SLURM_PROCID", "3")
    monkeypatch.setenv("SLURM_NTASKS", "8")
    monkeypatch.setenv("SLURM_LOCALID", "1")

    assert dist_utils.distributed_launch_detected()
    assert "RANK" not in os.environ
    assert "WORLD_SIZE" not in os.environ
    assert "LOCAL_RANK" not in os.environ

    dist_utils.normalize_slurm_environment()

    assert dist_utils.launch_contract() == "slurm-direct"
    assert dist_utils.distributed_launch_detected()
    assert os.environ["WORLD_SIZE"] == "8"
    assert os.environ["RANK"] == "3"
    assert dist_utils.distributed_local_rank() == 1
