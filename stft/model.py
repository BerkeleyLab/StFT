import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from stft.model_utils import TransformerLayer, get_2d_sincos_pos_embed


class SpectralPath(nn.Module):
    """FNO-style path of StFTBlock: lifts to frequency domain, attends, then reconstructs."""

    def __init__(
        self,
        cond_time,
        freq_in_channels,
        modes,
        out_channel,
        lift_channel=32,
        dim=256,
        depth=2,
        num_heads=1,
        mlp_dim=256,
        act="relu",
        grid_size=(4, 4),
        layer_indx=0,
    ):
        super().__init__()
        self.cond_time = cond_time
        self.freq_in_channels = freq_in_channels
        self.modes = modes
        self.out_channel = out_channel
        self.lift_channel = lift_channel
        self.layer_indx = layer_indx

        num_patches = grid_size[0] * grid_size[1]
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, dim), requires_grad=False
        )
        pos_embed = get_2d_sincos_pos_embed(dim, grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.p = nn.Linear(freq_in_channels, lift_channel)
        self.linear = nn.Linear(
            modes[0] * modes[1] * (cond_time + layer_indx) * lift_channel * 2,
            dim,
        )
        self.encoder_layers = nn.ModuleList(
            [TransformerLayer(dim, num_heads, mlp_dim, act) for _ in range(depth)]
        )
        self.q = nn.Linear(dim, modes[0] * modes[1] * 1 * lift_channel * 2)
        self.down = nn.Linear(lift_channel, out_channel)

    def forward(self, x):
        n, l, _, ph, pw = x.shape

        # Split into original time steps and added layer outputs
        x_or = x[:, :, : self.cond_time * self.freq_in_channels]
        x_added = x[:, :, (self.cond_time * self.freq_in_channels) :]
        x_or = rearrange(
            x_or,
            "n l (t v) ph pw -> n l ph pw t v",
            t=self.cond_time,
            v=self.freq_in_channels,
        )
        grid_dup = x_or[:, :, :, :, :1, -2:].repeat(1, 1, 1, 1, self.layer_indx, 1)
        x_added = rearrange(
            x_added,
            "n l (t v) ph pw -> n l ph pw t v",
            t=self.layer_indx,
            v=self.freq_in_channels - 2,
        )
        x_added = torch.cat((x_added, grid_dup), dim=-1)
        x = torch.cat((x_or, x_added), dim=-2)

        # Lift channels
        x = self.p(x)
        x = rearrange(x, "n l ph pw t v -> (n l) v t ph pw")

        # 3D rFFT, keep only low-frequency modes
        x_ft = torch.fft.rfftn(x, dim=[2, 3, 4])[
            :, :, :, : self.modes[0], : self.modes[1]
        ]
        x_ft_real = (x_ft.real).flatten(1)
        x_ft_imag = (x_ft.imag).flatten(1)
        x_ft_real = rearrange(x_ft_real, "(n l) D -> n l D", n=n, l=l)
        x_ft_imag = rearrange(x_ft_imag, "(n l) D -> n l D", n=n, l=l)
        x_ft_real_imag = torch.cat((x_ft_real, x_ft_imag), dim=-1)

        # Transformer in spectral token space
        x = self.linear(x_ft_real_imag)
        x = x + self.pos_embed
        for layer in self.encoder_layers:
            x = layer(x)

        # Decode back to spatial domain
        x_real, x_imag = self.q(x).split(
            self.modes[0] * self.modes[1] * self.lift_channel, dim=-1
        )
        x_real = x_real.reshape(n * l, -1, 1, self.modes[0], self.modes[1])
        x_imag = x_imag.reshape(n * l, -1, 1, self.modes[0], self.modes[1])
        x_complex = torch.complex(x_real, x_imag)
        out_ft = torch.zeros(
            n * l,
            self.lift_channel,
            1,
            ph,
            pw // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out_ft[:, :, :, : self.modes[0], : self.modes[1]] = x_complex
        x = torch.fft.irfftn(out_ft, s=(1, ph, pw))
        x = rearrange(x, "(n l) v t ph pw -> (n l) ph pw (v t)", n=n, l=l, t=1)
        x = self.down(x)
        return rearrange(x, "(n l) ph pw c -> n l c ph pw", n=n, l=l)


class SpatialPath(nn.Module):
    """ViT-style path of StFTBlock: patch tokens attended in spatial domain."""

    def __init__(
        self,
        in_dim,
        out_dim,
        out_channel,
        dim=256,
        depth=2,
        num_heads=1,
        mlp_dim=256,
        act="relu",
        grid_size=(4, 4),
    ):
        super().__init__()
        self.out_channel = out_channel

        num_patches = grid_size[0] * grid_size[1]
        self.token_embed = nn.Linear(in_dim, dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, dim), requires_grad=False
        )
        pos_embed = get_2d_sincos_pos_embed(dim, grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        self.encoder_layers = nn.ModuleList(
            [TransformerLayer(dim, num_heads, mlp_dim, act) for _ in range(depth)]
        )
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, out_dim))

    def forward(self, x):
        _, _, _, ph, pw = x.shape
        x = x.flatten(2)
        x = self.token_embed(x) + self.pos_embed
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.head(x)
        return rearrange(
            x, "n l (c ph pw) -> n l c ph pw", c=self.out_channel, ph=ph, pw=pw
        )


class StFTBlock(nn.Module):
    def __init__(
        self,
        cond_time,
        freq_in_channels,
        in_dim,
        out_dim,
        out_channel,
        modes,
        lift_channel=32,
        dim=256,
        depth=2,
        num_heads=1,
        mlp_dim=256,
        act="relu",
        grid_size=(4, 4),
        layer_indx=0,
    ):
        super().__init__()
        self.spectral = SpectralPath(
            cond_time=cond_time,
            freq_in_channels=freq_in_channels,
            modes=modes,
            out_channel=out_channel,
            lift_channel=lift_channel,
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            act=act,
            grid_size=grid_size,
            layer_indx=layer_indx,
        )
        self.spatial = SpatialPath(
            in_dim=in_dim,
            out_dim=out_dim,
            out_channel=out_channel,
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            act=act,
            grid_size=grid_size,
        )

    def forward(self, x):
        return self.spatial(x) + self.spectral(x)


class StFT(nn.Module):
    def __init__(
        self,
        cond_time,
        num_vars,
        patch_sizes,
        overlaps,
        in_channels,
        out_channels,
        modes,
        img_size=(50, 50),
        lift_channel=32,
        dim=128,
        vit_depth=3,
        num_heads=1,
        mlp_dim=128,
        act="relu",
        condition_blocks=True
    ):
        super().__init__()

        self.cond_time = cond_time
        self.num_vars = num_vars
        self.patch_sizes = patch_sizes
        self.overlaps = overlaps
        self.condition = condition_blocks

        blocks = []
        H, W = img_size
        for depth, (p1, p2) in enumerate(patch_sizes):
            overlap_h, overlap_w = overlaps[depth]
            step_h = p1 - overlap_h
            step_w = p2 - overlap_w
            pad_h = (step_h - (H - p1) % step_h) % step_h
            pad_w = (step_w - (W - p2) % step_w) % step_w
            num_patches_h = (H + pad_h - p1) // step_h + 1
            num_patches_w = (W + pad_w - p2) // step_w + 1

            in_dim = p1 * p2 * (in_channels if depth == 0 else in_channels + out_channels)
            blocks.append(
                StFTBlock(
                    cond_time,
                    num_vars,
                    in_dim,
                    out_channels * p1 * p2,
                    out_channels,
                    modes[depth],
                    lift_channel=lift_channel,
                    dim=dim,
                    depth=vit_depth[depth],
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    act=act,
                    grid_size=(num_patches_h, num_patches_w),
                    layer_indx=min(depth, 1),
                )
            )
        self.blocks = nn.ModuleList(blocks)

    @staticmethod
    def _patchify(x, patch_size, overlap):
        p1, p2 = patch_size
        overlap_h, overlap_w = overlap
        step_h = p1 - overlap_h
        step_w = p2 - overlap_w

        pad_h = (step_h - (x.shape[2] - p1) % step_h) % step_h
        pad_w = (step_w - (x.shape[3] - p2) % step_w) % step_w
        padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)

        x = F.pad(x, padding, mode="constant", value=0)
        _, _, H_pad, W_pad = x.shape
        h = (H_pad - p1) // step_h + 1
        w = (W_pad - p2) // step_w + 1

        patches = x.unfold(2, p1, step_h).unfold(3, p2, step_w)
        patches = rearrange(patches, "n c h w ph pw -> n (h w) c ph pw")
        return patches, (p1, p2, step_h, step_w, padding, H_pad, W_pad, h, w)

    @staticmethod
    def _unpatchify(patches, restore_params):
        p1, p2, step_h, step_w, padding, H_pad, W_pad, h, w = restore_params

        patches = rearrange(patches, "n (h w) c ph pw -> n c h w ph pw", h=h, w=w)
        output = F.fold(
            rearrange(patches, "n c h w ph pw -> n (c ph pw) (h w)"),
            output_size=(H_pad, W_pad),
            kernel_size=(p1, p2),
            stride=(step_h, step_w),
        )
        overlap_count = F.fold(
            rearrange(
                torch.ones_like(patches), "n c h w ph pw -> n (c ph pw) (h w)"
            ),
            output_size=(H_pad, W_pad),
            kernel_size=(p1, p2),
            stride=(step_h, step_w),
        )
        output = output / overlap_count
        output = output[
            :, :, padding[2] : H_pad - padding[3], padding[0] : W_pad - padding[1]
        ]
        return output

    def forward(self, x, grid):
        grid_dup = grid.unsqueeze(0).expand(x.shape[0], x.shape[1], -1, -1, -1)
        x = torch.cat((x, grid_dup), dim=2)
        x = rearrange(x, "B L C H W -> B (L C) H W")

        layer_outputs = []
        for depth, block in enumerate(self.blocks):
            inputs = torch.cat((x, layer_outputs[-1].detach().clone()), dim=1) if (depth != 0 and self.condition) else x
            patches, restore_params = self._patchify(
                inputs,
                self.patch_sizes[depth],
                self.overlaps[depth],
            )
            output = block(patches)
            layer_outputs.append(self._unpatchify(output, restore_params))

        return layer_outputs