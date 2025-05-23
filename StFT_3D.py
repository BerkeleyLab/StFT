import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import TransformerLayer, get_2d_sincos_pos_embed


class StFTBlcok(nn.Module):
    def __init__(
        self,
        cond_time,
        freq_in_channels,
        in_dim,
        out_dim,
        out_channel,
        num_patches,
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
        super(StFTBlcok, self).__init__()
        self.layer_indx = layer_indx
        self.cond_time = cond_time
        self.freq_in_channels = freq_in_channels
        self.modes = modes
        self.out_channel = out_channel
        self.lift_channel = lift_channel
        self.token_embed = nn.Linear(in_dim, dim)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, dim), requires_grad=False
        )
        self.pos_embed_f = nn.Parameter(
            torch.randn(1, num_patches, dim), requires_grad=False
        )
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], grid_size)
        pos_embed_f = get_2d_sincos_pos_embed(self.pos_embed_f.shape[-1], grid_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        self.pos_embed_f.data.copy_(
            torch.from_numpy(pos_embed_f).float().unsqueeze(0)
        )
        self.encoder_layers = nn.ModuleList(
            [TransformerLayer(dim, num_heads, mlp_dim, act) for _ in range(depth)]
        )
        self.encoder_layers_f = nn.ModuleList(
            [TransformerLayer(dim, num_heads, mlp_dim, act) for _ in range(depth)]
        )
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, out_dim))
        self.p = nn.Linear(freq_in_channels, lift_channel)
        self.linear = nn.Linear(
            modes[0] * modes[1] * (self.cond_time + self.layer_indx) * lift_channel * 2,
            dim,
        )
        self.q = nn.Linear(dim, modes[0] * modes[1] * 1 * lift_channel * 2)
        self.down = nn.Linear(lift_channel, out_channel)

    def forward(self, x):
        x_copy = x
        n, l, _, ph, pw = x.shape
        x_or = x[:, :, : self.cond_time * self.freq_in_channels]
        x_added = x[:, :, (self.cond_time * self.freq_in_channels) :]
	x_or = x_or.permute(0, 1, 3, 4, 2).view(n, l, ph, pw, self.cond_time, self.freq_in_channels)
        grid_dup = x_or[:, :, :, :, :1, -2:].repeat(1, 1, 1, 1, self.layer_indx, 1)

	x_added = x_added.permute(0, 1, 3, 4, 2).view(n, l, ph, pw, self.layer_indx, self.freq_in_channels - 2)
        x_added = torch.cat((x_added, grid_dup), axis=-1)
        x = torch.cat((x_or, x_added), axis=-2)
        x = self.p(x)
	v, t = x.shape[-1], x.shape[-2]
	x = x.permute(0, 1, 5, 4, 2, 3).view(n * l, v, t, ph, pw)
        x_ft = torch.fft.rfftn(x, dim=[2, 3, 4])[
            :, :, :, : self.modes[0], : self.modes[1]
        ]
        x_ft_real = (x_ft.real).flatten(1)
        x_ft_imag = (x_ft.imag).flatten(1)
	x_ft_real = x_ft_real.view(n, l, -1)
	x_ft_imag = x_ft_imag.view(n, l, -1)
        x_ft_real_imag = torch.cat((x_ft_real, x_ft_imag), axis=-1)
        x = self.linear(x_ft_real_imag)
        x = x + self.pos_embed_f
        for layer in self.encoder_layers_f:
            x = layer(x)
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
	x = x.permute(0, 3, 4, 1, 2).view(n * l, ph, pw, -1)
        x = self.down(x)
	c = x.shape[-1]
	x_f = x.permute(0, 3, 1, 2).view(n, l, c, ph, pw)
        x = x_copy
        _, _, _, ph, pw = x.shape
        x = x.flatten(2)
        x = self.token_embed(x) + self.pos_embed
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.head(x)
	x = x.view(n, l, self.out_channel, ph, pw)
        x = x + x_f
        return x


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
    ):
        super(StFT, self).__init__()

        blocks = []
        self.cond_time = cond_time
        self.num_vars = num_vars
        self.patch_sizes = patch_sizes
        self.overlaps = overlaps
        for depth, (p1, p2) in enumerate(patch_sizes):
            H, W = img_size
            cur_modes = modes[depth]
            cur_depth = vit_depth[depth]
            overlap_h, overlap_w = overlaps[depth]

            step_h = p1 - overlap_h
            step_w = p2 - overlap_w

            pad_h = (step_h - (H - p1) % step_h) % step_h
            pad_w = (step_w - (W - p2) % step_w) % step_w
            H_pad = H + pad_h
            W_pad = W + pad_w

            num_patches_h = (H_pad - p1) // step_h + 1
            num_patches_w = (W_pad - p2) // step_w + 1

            num_patches = num_patches_h * num_patches_w
            if depth == 0:
                blocks.append(
                    StFTBlcok(
                        cond_time,
                        num_vars,
                        p1 * p2 * in_channels,
                        out_channels * p1 * p2,
                        out_channels,
                        num_patches,
                        cur_modes,
                        lift_channel=lift_channel,
                        dim=dim,
                        depth=cur_depth,
                        num_heads=num_heads,
                        mlp_dim=mlp_dim,
                        act=act,
                        grid_size=(num_patches_h, num_patches_w),
                        layer_indx=depth,
                    )
                )
            else:
                blocks.append(
                    StFTBlcok(
                        cond_time,
                        num_vars,
                        p1 * p2 * (in_channels + out_channels),
                        out_channels * p1 * p2,
                        out_channels,
                        num_patches,
                        cur_modes,
                        lift_channel=lift_channel,
                        dim=dim,
                        depth=cur_depth,
                        num_heads=num_heads,
                        mlp_dim=mlp_dim,
                        act=act,
                        grid_size=(num_patches_h, num_patches_w),
                        layer_indx=1,
                    )
                )

        self.blocks = nn.ModuleList(blocks)

    def forward(self, x, grid):
        grid_dup = grid[None, :, :, :].repeat(x.shape[0], x.shape[1], 1, 1, 1)
        x = torch.cat((x, grid_dup), axis=2)
	x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
        layer_outputs = []
        patches = x
        restore_params = []
        or_patches = x
        if True:
            for depth in range(len(self.patch_sizes)):
                if True:
                    p1, p2 = self.patch_sizes[depth]
                    overlap_h, overlap_w = self.overlaps[depth]

                    step_h = p1 - overlap_h
                    step_w = p2 - overlap_w

                    pad_h = (step_h - (patches.shape[2] - p1) % step_h) % step_h
                    pad_w = (step_w - (patches.shape[3] - p2) % step_w) % step_w
                    padding = (
                        pad_w // 2,
                        pad_w - pad_w // 2,
                        pad_h // 2,
                        pad_h - pad_h // 2,
                    )

                    patches = F.pad(patches, padding, mode="constant", value=0)
                    _, _, H_pad, W_pad = patches.shape

                    h = (H_pad - p1) // step_h + 1
                    w = (W_pad - p2) // step_w + 1

                    restore_params.append(
                        (p1, p2, step_h, step_w, padding, H_pad, W_pad, h, w)
                    )

                    patches = patches.unfold(2, p1, step_h).unfold(3, p2, step_w)
	            n, c, h, w, ph, pw = x.shape		
		    patches = patches.permute(0, 2, 3, 1, 4, 5).view(n, h*w, c, ph, pw)
                    processed_patches = self.blocks[depth](patches)

		    patches = processed_patches.permute(0, 2, 1, 3, 4).view(n, c, h, w, ph, pw)
                    output = F.fold(
			torch.reshape(patches.permute(0, 1, 4, 5, 2, 3),(n, c * ph * pw, h * w)),
                        output_size=(H_pad, W_pad),
                        kernel_size=(p1, p2),
                        stride=(step_h, step_w),
                    )

                    overlap_count = F.fold(
			torch.reshape(torch.ones_like(patches).permute(0, 1, 4, 5, 2, 3),(n, c * ph * pw, h * w)),
                        output_size=(H_pad, W_pad),
                        kernel_size=(p1, p2),
                        stride=(step_h, step_w),
                    )
                    output = output / overlap_count
                    output = output[
                        :,
                        :,
                        padding[2] : H_pad - padding[3],
                        padding[0] : W_pad - padding[1],
                    ]
                    layer_outputs.append(output)
                    added = output
                    patches = torch.cat((or_patches, added.detach().clone()), axis=1)

        return layer_outputs
