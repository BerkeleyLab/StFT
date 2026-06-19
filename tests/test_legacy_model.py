import torch
import torch.nn as nn

import trainer as trainer_module
from stft.legacy import LegacyStFTAdapter
from trainer import Trainer


class RecordingLegacyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.seen = None

    def forward(self, x):
        self.seen = x
        return [x[:, :1] * self.weight]


def test_legacy_adapter_uses_legacy_channel_layout():
    legacy_model = RecordingLegacyModel()
    adapter = LegacyStFTAdapter(legacy_model)
    x = torch.randn(2, 3, 1, 4, 5)
    grid = torch.randn(2, 4, 5)

    outputs = adapter(x, grid)

    assert len(outputs) == 1
    assert legacy_model.seen.shape == (2, 5, 4, 5)
    assert torch.equal(legacy_model.seen[:, :3], x.reshape(2, 3, 4, 5))
    assert torch.equal(legacy_model.seen[:, 3:], grid.unsqueeze(0).expand(2, -1, -1, -1))


def test_build_legacy_model_uses_legacy_dimensions(monkeypatch):
    captured = {}

    def fake_build_legacy_hierarrm(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return RecordingLegacyModel()

    monkeypatch.setattr(
        trainer_module,
        "build_legacy_hierarrm",
        fake_build_legacy_hierarrm,
    )

    trainer = Trainer.__new__(Trainer)
    trainer.condition = True
    trainer.legacy_config = {"modes": 2, "vit_depth": 1}
    trainer.num_in_states = 1
    trainer.cond_time = 3
    trainer.patch_sizes = ((4, 4),)
    trainer.overlaps = ((1, 1),)
    trainer.img_size = (8, 8)
    trainer.dim = 8
    trainer.num_heads = 1
    trainer.modes = ((2, 2),)
    trainer.vit_depth = (1,)

    model = trainer.build_legacy_model(dim=8)

    assert isinstance(model, LegacyStFTAdapter)
    assert captured["args"] == (
        trainer.patch_sizes,
        trainer.overlaps,
        trainer.num_in_states * trainer.cond_time + 2,
        trainer.num_in_states,
    )
    assert captured["kwargs"] == {
        "img_size": trainer.img_size,
        "dim": 8,
        "vit_depth": 1,
        "modes": 2,
        "num_heads": trainer.num_heads,
        "mlp_dim": 8,
    }
