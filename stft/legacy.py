import torch
import torch.nn as nn
from einops import rearrange


class LegacyStFTAdapter(nn.Module):
    """Adapt legacy 2D StFT models to the current Trainer model API."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, grid):
        batch_size = x.shape[0]
        x = rearrange(x, "b t c h w -> b (t c) h w")
        grid = grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
        x = torch.cat((x, grid), dim=1)
        return self.model(x)


def build_legacy_hierarrm(*args, **kwargs):
    from SWE.sw_2d import HierARRM

    return HierARRM(*args, **kwargs)
