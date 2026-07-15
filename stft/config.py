from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf


IMMUTABLE_CONFIG_KEYS = (
    "model_type",
    "legacy",
    "dataset",
    "patch_sizes",
    "overlaps",
    "vit_depth",
    "modes",
    "dim",
    "num_heads",
    "seed",
    "cond_time",
    "lift_channel",
    "act",
    "condition_blocks",
    "use_snapshots",
    "snapshot_length",
)

WARN_CONFIG_KEYS = (
    "lr",
    "batchsize",
)


def to_plain_config(config: Any) -> dict[str, Any]:
    """Convert Hydra/OmegaConf or Python config objects to YAML-safe containers."""
    if isinstance(config, DictConfig):
        container = OmegaConf.to_container(config, resolve=True)
    else:
        container = config
    plain = _normalize_container(container)
    if not isinstance(plain, dict):
        raise TypeError(f"config must resolve to a dict, got {type(plain).__name__}")
    plain.pop("hydra", None)
    return plain


def load_config(path: str | Path) -> dict[str, Any]:
    return to_plain_config(OmegaConf.load(path))


def save_run_config(config: dict[str, Any], save_path: str | Path | None = None) -> None:
    plain = to_plain_config(config)
    run_dir = Path(save_path or plain["save_path"])
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"

    if config_path.exists():
        existing = load_config(config_path)
        validate_resume_config(existing, plain)
        warn_config_changes(existing, plain)

    OmegaConf.save(config=OmegaConf.create(plain), f=config_path)


def validate_resume_config(
    existing: dict[str, Any],
    current: dict[str, Any],
    immutable_keys: tuple[str, ...] = IMMUTABLE_CONFIG_KEYS,
) -> None:
    mismatches = [
        (key, existing.get(key), current.get(key))
        for key in immutable_keys
        if _normalize_container(existing.get(key)) != _normalize_container(current.get(key))
    ]
    if not mismatches:
        return

    details = "\n".join(
        f"  {key}: existing={old!r}, current={new!r}"
        for key, old, new in mismatches
    )
    raise ValueError(
        "Refusing to resume run with changed immutable config values:\n"
        f"{details}"
    )


def warn_config_changes(
    existing: dict[str, Any],
    current: dict[str, Any],
    warn_keys: tuple[str, ...] = WARN_CONFIG_KEYS,
) -> list[str]:
    warnings = []
    for key in warn_keys:
        old = _normalize_container(existing.get(key))
        new = _normalize_container(current.get(key))
        if old != new:
            warnings.append(
                f"Config value changed for resumable run: {key}: existing={old!r}, current={new!r}"
            )
    for warning in warnings:
        print(f"WARNING: {warning}", flush=True)
    return warnings


def resolve_run_dir(
    *,
    run: int | str | None = None,
    run_dir: str | Path | None = None,
    experiments_dir: str | Path = "experiments",
) -> Path:
    if run_dir is not None:
        return Path(run_dir)
    if run is None:
        raise ValueError("Either run or run_dir must be provided")
    run_text = str(run)
    if not run_text.startswith("run_"):
        run_text = f"run_{run_text}"
    return Path(experiments_dir) / run_text


def load_run_config(
    *,
    run: int | str | None = None,
    run_dir: str | Path | None = None,
    experiments_dir: str | Path = "experiments",
) -> dict[str, Any]:
    resolved_run_dir = resolve_run_dir(
        run=run,
        run_dir=run_dir,
        experiments_dir=experiments_dir,
    )
    config_path = resolved_run_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing run config: {config_path}")
    config = load_config(config_path)
    config["save_path"] = str(resolved_run_dir)
    return config


def _normalize_container(value: Any) -> Any:
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, dict):
        return {
            str(key): _normalize_container(item)
            for key, item in value.items()
            if key != "hydra"
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_container(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
