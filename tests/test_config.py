import pytest

from stft.config import load_config, load_run_config, save_run_config


def make_config(save_path):
    return {
        "dataset": "/data/shallow-water",
        "patch_sizes": [[128, 128], [64, 64]],
        "overlaps": [[1, 1], [1, 1]],
        "vit_depth": [6, 6],
        "modes": [[8, 8], [8, 8]],
        "dim": 512,
        "num_heads": 1,
        "lr": 1e-4,
        "max_epochs": 10000,
        "batchsize": 16,
        "cond_time": 5,
        "lift_channel": 64,
        "act": "gelu",
        "save_path": str(save_path),
        "save_every_n": 5,
        "validate_every_n": 10,
        "condition_blocks": False,
        "use_snapshots": True,
        "snapshot_length": 20,
    }


def test_save_run_config_writes_resolved_config(tmp_path):
    run_dir = tmp_path / "experiments" / "run_2"
    config = make_config(run_dir)

    save_run_config(config)

    saved = load_config(run_dir / "config.yaml")
    assert saved == config


def test_save_run_config_allows_mutable_resume_change(tmp_path):
    run_dir = tmp_path / "experiments" / "run_2"
    config = make_config(run_dir)
    save_run_config(config)

    changed = {**config, "max_epochs": 20000}
    save_run_config(changed)

    saved = load_config(run_dir / "config.yaml")
    assert saved["max_epochs"] == 20000


def test_save_run_config_rejects_immutable_resume_change(tmp_path):
    run_dir = tmp_path / "experiments" / "run_2"
    config = make_config(run_dir)
    save_run_config(config)

    changed = {**config, "cond_time": 6}
    with pytest.raises(ValueError, match="immutable config"):
        save_run_config(changed)


def test_save_run_config_warns_for_sensitive_resume_change(tmp_path, capsys):
    run_dir = tmp_path / "experiments" / "run_2"
    config = make_config(run_dir)
    save_run_config(config)

    changed = {**config, "lr": 3e-4}
    save_run_config(changed)

    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "lr" in captured.out


def test_load_run_config_resolves_run_number(tmp_path):
    experiments_dir = tmp_path / "experiments"
    run_dir = experiments_dir / "run_2"
    config = make_config(run_dir)
    save_run_config(config)

    loaded = load_run_config(run=2, experiments_dir=experiments_dir)

    assert loaded["save_path"] == str(run_dir)
    assert loaded["cond_time"] == config["cond_time"]
