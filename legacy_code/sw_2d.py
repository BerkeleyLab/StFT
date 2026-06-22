import torch
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange
import math
from torch.utils.data import Dataset
import random
import numpy as np

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)
    
    
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.qkv = nn.Linear(dim, dim * 3)
        self.fc_out = nn.Linear(dim, dim)

    def forward(self, x):
        batch_size, seq_length, dim = x.size()
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
        attn_output = attn_output.view(batch_size, seq_length, dim)

        out = self.fc_out(attn_output)
        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(dim, num_heads)
        self.feed_forward = FeedForward(dim, mlp_dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_output = self.self_attn(self.ln1(x))
        x = x + attn_output
        ff_output = self.feed_forward(self.ln2(x))
        x = x + ff_output
        return x


class HLayer(nn.Module):
    def __init__(
        self,
        fno_in_channels,
        in_dim,
        out_dim,
        out_channel,
        num_patches,
        dim=256,
        depth=2,
        num_heads=1,
        mlp_dim=256,
        modes=8,
    ):
        super(HLayer, self).__init__()
        self.modes=modes
        self.fno_in_channels = fno_in_channels
        self.out_channel = out_channel
        self.token_embed = nn.Linear(in_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, dim))
        self.pos_embed_fno = nn.Parameter(torch.randn(1, num_patches, dim))
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(dim, num_heads, mlp_dim) for _ in range(depth)]
        )
        
        self.encoder_layers_fno = nn.ModuleList(
            [TransformerEncoderLayer(dim, num_heads, mlp_dim) for _ in range(depth)]
        )
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, out_dim))
        self.p = nn.Linear(fno_in_channels, 32)
        
        
        self.linear = nn.Linear(modes*modes*32*2, dim)
        
        self.q = nn.Linear(dim, modes*modes*32*2)
        
        self.down = nn.Linear(32, out_channel)

    def forward(self, x):
        x_copy = x 
        n, l, _, ph, pw = x.shape
        x = rearrange(x, "n l c ph pw -> n l ph pw c")
        x = self.p(x)
        x = rearrange(x, "n l ph pw c -> (n l) c ph pw")
        x_ft = torch.fft.rfft2(x)[:,:,:self.modes,:self.modes]
        x_ft_real = (x_ft.real).flatten(1)
        x_ft_imag = (x_ft.imag).flatten(1)
        x_ft_real = rearrange(x_ft_real, "(n l) D -> n l D",n=n,l=l)
        x_ft_imag = rearrange(x_ft_imag, "(n l) D -> n l D",n=n,l=l)
        x_ft_real_imag = torch.cat((x_ft_real,x_ft_imag),axis=-1) 
        x = self.linear(x_ft_real_imag)
        x = x + self.pos_embed_fno
        for layer in self.encoder_layers_fno:
            x = layer(x)
        x_real, x_imag = self.q(x).split(self.modes*self.modes*32,dim=-1)
        x_real=x_real.reshape(n*l,-1,self.modes,self.modes)
        x_imag=x_imag.reshape(n*l,-1,self.modes,self.modes)
        x_complex = torch.complex(x_real,x_imag)
        out_ft = torch.zeros(n*l, 32,  ph, pw//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes, :self.modes] = x_complex
        x = torch.fft.irfft2(out_ft, s=(ph,pw))
        x = rearrange(x, "(n l) c ph pw -> (n l) ph pw c",n=n,l=l)
        x = self.down(x)
        x_fno = rearrange(x, "(n l) ph pw c -> n l c ph pw",n=n,l=l)        
        x = x_copy
        _, _, _, ph, pw = x.shape
        x = x.flatten(2)
        x = self.token_embed(x) + self.pos_embed
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.head(x)
        x = rearrange(
            x, "n l (c ph pw) -> n l c ph pw", c=self.out_channel, ph=ph, pw=pw
        )
        x = x + x_fno
    
        return x


class HierARRM(nn.Module):
    def __init__(
        self,
        patch_sizes,
        overlaps,
        in_channels,
        out_channels,
        img_size=(50, 50),
        dim=128,
        vit_depth=3,
        modes = 8,
        num_heads=1,
        mlp_dim=128,
        ):
        super(HierARRM, self).__init__()

        hlayers = []
        self.patch_sizes = patch_sizes
        self.overlaps = overlaps
        for depth, (p1, p2) in enumerate(patch_sizes):
            if depth == 0:
                H, W = img_size
            else:
                H, W = img_size

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
            # modes=None
            if p1//2 <= modes:
                modes = p1//2
            # else:
            #     modes = modes
            if depth == 0:
                hlayers.append(
                    HLayer(
                        in_channels,
                        p1 * p2 * in_channels,
                        out_channels * p1 * p2,
                        out_channels,
                        num_patches,
                        dim=dim,
                        depth=vit_depth,
                        num_heads=num_heads,
                        mlp_dim=mlp_dim,
                        modes=modes,
                    )
                )
                print(p1 * p2 * in_channels, out_channels * p1 * p2, num_patches)
            else:
                hlayers.append(
                    HLayer(
                        in_channels + out_channels,
                        p1 * p2 * (in_channels + out_channels),
                        out_channels * p1 * p2,
                        out_channels,
                        num_patches,
                        dim=dim,
                        depth=vit_depth,
                        num_heads=num_heads,
                        mlp_dim=mlp_dim,
                        modes=modes,
                    )
                )
                print(
                    p1 * p2 * (in_channels + out_channels),
                    out_channels * p1 * p2,
                    num_patches,
                )

        self.hlayers = nn.ModuleList(hlayers)

    def forward(self, x):
        layer_outputs = []
        patches = x

        restore_params = []

        or_patches = x

        if True:

            for depth in range(len(self.patch_sizes)):
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
                patches = rearrange(patches, "n c h w ph pw -> n (h w) c ph pw")
                processed_patches = self.hlayers[depth](patches)

                patches = rearrange(
                    processed_patches, "n (h w) c ph pw -> n c h w ph pw", h=h, w=w
                )

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
                    :,
                    :,
                    padding[2] : H_pad - padding[3],
                    padding[0] : W_pad - padding[1],
                ]
                layer_outputs.append(output)
                patches = torch.cat((or_patches, output.detach().clone()), axis=1)

       
        return layer_outputs



class TemporalDataset(Dataset):
    def __init__(self, data, snapshot_length=20):
        self.data = data 
        self.N, self.T, self.C, self.H, self.W = data.shape
        self.snapshot_length = snapshot_length

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        start = random.randint(0, self.T - self.snapshot_length)
        selected_data = self.data[idx, start:start + self.snapshot_length]
        return selected_data
    


def get_grid(H,W):
    x = np.linspace(0,1,H)
    y = np.linspace(0,1,W)
    
    x,y = np.meshgrid(x,y)
    x=x.T
    y=y.T
    
    grid = torch.tensor(np.concatenate((x[None],y[None]),axis=0),dtype=torch.float32)
    
    return grid