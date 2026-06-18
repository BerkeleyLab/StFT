import json
from datetime import datetime, timezone

import torch

from stft import distributed as dist_utils
from stft.launch_metadata import (
    SLURM_ENV_KEYS,
    collect_launch_metadata,
    write_launch_metadata,
)
import trainer as trainer_module
from trainer import Trainer


_DIST_ENV_VARS = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
)


def test_collect_launch_metadata_captures_slurm_and_observed_state(monkeypatch):
    for env_name in (*SLURM_ENV_KEYS.values(), *_DIST_ENV_VARS):
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setattr(dist_utils, "DDP_LAUNCHER", None)
    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    monkeypatch.setenv("SLURM_NNODES", "2")
    monkeypatch.setenv("SLURM_NTASKS", "8")
    monkeypatch.setenv("SLURM_GPUS_PER_NODE", "4")
    monkeypatch.setenv("SLURM_PROCID", "3")
    monkeypatch.setenv("SLURM_LOCALID", "1")
    monkeypatch.setenv("MASTER_ADDR", "nid000001")
    monkeypatch.setenv("MASTER_PORT", "29500")

    timestamp = datetime(2026, 6, 18, 12, 3, 4, tzinfo=timezone.utc)
    metadata = collect_launch_metadata(
        device=torch.device("cpu"),
        local_rank=1,
        rank=3,
        world_size=8,
        start_epoch=12,
        resume_from_checkpoint=True,
        timestamp=timestamp,
    )

    assert metadata["timestamp"] == "2026-06-18T12:03:04Z"
    assert metadata["slurm"]["job_id"] == "123456"
    assert metadata["slurm"]["nnodes"] == "2"
    assert metadata["slurm"]["ntasks"] == "8"
    assert metadata["slurm"]["gpus_per_node"] == "4"
    assert metadata["torch_distributed"]["world_size"] == 8
    assert metadata["torch_distributed"]["rank"] == 3
    assert metadata["torch_distributed"]["local_rank"] == 1
    assert metadata["torch_distributed"]["master_addr"] == "nid000001"
    assert metadata["torch_distributed"]["master_port"] == "29500"
    assert metadata["runtime"]["device"] == "cpu"
    assert metadata["training"]["start_epoch"] == 12
    assert metadata["training"]["resume_from_checkpoint"] is True


def test_write_launch_metadata_creates_unique_append_only_records(tmp_path, monkeypatch):
    for env_name in SLURM_ENV_KEYS.values():
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    timestamp = datetime(2026, 6, 18, 12, 3, 4, 123456, tzinfo=timezone.utc)

    first_path, first_metadata = write_launch_metadata(
        tmp_path,
        device=torch.device("cpu"),
        local_rank=0,
        rank=0,
        world_size=1,
        timestamp=timestamp,
    )
    second_path, second_metadata = write_launch_metadata(
        tmp_path,
        device=torch.device("cpu"),
        local_rank=0,
        rank=0,
        world_size=1,
        timestamp=timestamp,
    )

    assert first_path != second_path
    assert first_path.exists()
    assert second_path.exists()
    assert first_path.parent.name == first_metadata["launch_id"]
    assert second_path.parent.name == second_metadata["launch_id"]
    assert second_path.parent.name.endswith("-2")

    first_saved = json.loads(first_path.read_text())
    second_saved = json.loads(second_path.read_text())
    assert first_saved["launch_id"] == first_path.parent.name
    assert second_saved["launch_id"] == second_path.parent.name


def test_trainer_records_launch_metadata_to_wandb_summary(tmp_path, monkeypatch):
    launch_path = tmp_path / "launches" / "launch-1" / "launch.json"
    metadata = {
        "slurm": {
            "job_id": "123456",
            "nnodes": "2",
            "job_num_nodes": None,
            "gpus_per_node": "4",
        },
        "torch_distributed": {"world_size": 8},
        "runtime": {"cuda_device_count": 4},
    }
    summary = {}
    saved = []

    def fake_write_launch_metadata(save_path, **kwargs):
        assert save_path == tmp_path
        assert kwargs["local_rank"] == 0
        assert kwargs["rank"] == 0
        assert kwargs["world_size"] == 8
        assert kwargs["start_epoch"] == 3
        assert kwargs["resume_from_checkpoint"] is True
        return launch_path, metadata

    def fake_wandb_save(path, base_path=None):
        saved.append((path, base_path))

    monkeypatch.setattr(
        trainer_module,
        "write_launch_metadata",
        fake_write_launch_metadata,
    )
    monkeypatch.setattr(trainer_module.wandb, "save", fake_wandb_save)
    monkeypatch.setattr(trainer_module.wandb, "summary", summary)

    trainer = Trainer.__new__(Trainer)
    trainer.is_main = True
    trainer.save_path = tmp_path
    trainer.device = torch.device("cpu")
    trainer.local_rank = 0
    trainer.rank = 0
    trainer.world_size = 8
    trainer.start_epoch = 3

    trainer.record_launch_metadata()

    assert saved == [(str(launch_path), str(tmp_path))]
    assert summary["latest_launch/job_id"] == "123456"
    assert summary["latest_launch/nnodes"] == "2"
    assert summary["latest_launch/world_size"] == 8
    assert summary["latest_launch/gpus_per_node"] == "4"


def test_trainer_reads_wandb_run_id_from_json(tmp_path):
    run_file = tmp_path / "wandb_run.json"
    run_file.write_text(
        json.dumps(
            {
                "id": "abc123",
                "name": "robust-dew-6",
                "url": "https://wandb.ai/entity/stft/runs/abc123",
            }
        )
    )

    trainer = Trainer.__new__(Trainer)
    trainer.save_path = tmp_path

    assert trainer._get_wandb_run_id() == "abc123"


def test_trainer_creates_wandb_run_json(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer_module.wandb.util, "generate_id", lambda: "new123")

    trainer = Trainer.__new__(Trainer)
    trainer.save_path = tmp_path

    assert trainer._get_wandb_run_id() == "new123"
    assert json.loads((tmp_path / "wandb_run.json").read_text()) == {
        "id": "new123",
        "name": None,
        "url": None,
    }


def test_trainer_migrates_legacy_wandb_run_id_file(tmp_path):
    legacy_file = tmp_path / "wandb_run_id.txt"
    legacy_file.write_text("legacy123\n")

    trainer = Trainer.__new__(Trainer)
    trainer.save_path = tmp_path

    assert trainer._get_wandb_run_id() == "legacy123"
    assert not legacy_file.exists()
    assert json.loads((tmp_path / "wandb_run.json").read_text()) == {
        "id": "legacy123",
        "name": None,
        "url": None,
    }


def test_trainer_saves_wandb_run_name_and_url(tmp_path):
    trainer = Trainer.__new__(Trainer)
    trainer.save_path = tmp_path
    trainer._wandb_run = type(
        "FakeWandbRun",
        (),
        {
            "id": "abc123",
            "name": "robust-dew-6",
            "url": "https://wandb.ai/entity/stft/runs/abc123",
        },
    )()

    trainer._save_wandb_run_metadata()

    assert json.loads((tmp_path / "wandb_run.json").read_text()) == {
        "id": "abc123",
        "name": "robust-dew-6",
        "url": "https://wandb.ai/entity/stft/runs/abc123",
    }
