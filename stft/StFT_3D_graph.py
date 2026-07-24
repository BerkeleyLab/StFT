"""StFT with a graph-based channel (GvT-style) for irregular domains.

Adds a third parallel branch to every StFTBlock, alongside the ViT branch and
the Fourier branch. Within each patch, the field is split into small
sub-patches; each sub-patch is a graph node whose PHYSICAL centroid coordinates
are read off the grid channels that already flow through the block. Following
"Graph-based vision transformer with sparsity" (GvT, Sci. Rep. 2025):

  adjacency  A = (A_I + I) ⊙ E                      (Eq. 3)
  Q/K        x_qk = (D^-1/2 A D^-1/2) x W_qk        (Eq. 4)
  attention  S = softmax(q k^T / sqrt(d))           (Eq. 5)
  values     x_v = (D^-1/2 R D^-1/2) x W_v + x      (Eq. 8)

with two adaptations for the irregular setting:
  * A_I is a Gaussian-kernel k-NN graph in TRUE physical coordinates instead
    of 8-directional index neighbors — this is where the real (non-uniform)
    geometry enters the model, which the FFT branch cannot see.
  * Talking-heads sparse selection is omitted (R = S): with ~64 nodes and
    dim >= 64 per head there is no low-rank bottleneck to fix. The
    normalization hook is kept so it can be added later.

Note: zero-padded border sub-patches get centroid coords ~0 and all-zero
features; they participate in the graph but carry no signal.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .StFT_3D import StFTBlock, StFT
from .model_utils import FeedForward


def sym_norm(A, eps=1e-6):
    """D^-1/2 A D^-1/2 (GCN symmetric normalization; keeps spectral radius <= ~1,
    unlike D^-1/2 A D^+1/2 which is merely a similarity transform of A)."""
    d = A.sum(-1).clamp_min(eps)
    dis = d.pow(-0.5)
    return A * dis.unsqueeze(-1) * dis.unsqueeze(-2)


class GraphAttentionLayer(nn.Module):
    """One GvT residual layer: graph-conv Q/K -> dot-product attention ->
    graph-conv values on the relation matrix, then a feed-forward block."""

    def __init__(self, dim, num_heads, mlp_dim, act="gelu"):
        super(GraphAttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim
        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.feed_forward = FeedForward(dim, mlp_dim, act)

    def _split(self, x):
        return rearrange(x, "b m (h d) -> b h m d", h=self.num_heads)

    def forward(self, x, A_hat):
        # x: (B, m, dim); A_hat: (B, m, m) normalized adjacency
        h = self.ln1(x)
        hg = torch.bmm(A_hat, h)  # Eq. 4: graph-conv aggregation for Q/K
        q = self._split(self.wq(hg))
        k = self._split(self.wk(hg))
        S = F.softmax(
            torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim**0.5), dim=-1
        )  # Eq. 5
        # Eq. 8 with R = S (talking-heads omitted). sym_norm is an identity for
        # row-stochastic S but is kept so a future relation matrix R != S slots in.
        R_hat = sym_norm(S)
        v = self._split(h)
        out = rearrange(torch.matmul(R_hat, v), "b h m d -> b m (h d)")
        x = x + self.wv(out)  # aggregate-then-project; residual absorbs Eq. 8's +x
        x = x + self.feed_forward(self.ln2(x))
        return x


class GraphBranch(nn.Module):
    """Graph channel of one StFTBlock: tokenize each patch into sub-patches,
    build a physical-coordinate-aware adjacency, run GvT layers, project back."""

    def __init__(
        self,
        per_pixel_channels,
        out_channel,
        freq_in_channels,
        patch_size,
        graph_patch,
        dim=128,
        depth=2,
        num_heads=1,
        mlp_dim=128,
        act="gelu",
        knn_k=8,
    ):
        super(GraphBranch, self).__init__()
        p1, p2 = patch_size
        g1, g2 = graph_patch
        assert p1 % g1 == 0 and p2 % g2 == 0, "graph_patch must divide patch_size"
        self.g1, self.g2 = g1, g2
        self.out_channel = out_channel
        self.freq_in_channels = freq_in_channels
        self.knn_k = knn_k
        self.embed = nn.Linear(g1 * g2 * per_pixel_channels, dim)
        self.layers = nn.ModuleList(
            [GraphAttentionLayer(dim, num_heads, mlp_dim, act) for _ in range(depth)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, g1 * g2 * out_channel)
        )

    def _adjacency(self, tok, coords, n):
        """A = (A_I + I) ⊙ E, symmetrically normalized.

        A_I: Gaussian-kernel k-NN graph in physical coordinates (per patch
             position, shared across the batch since the grid is static).
        E:   row-softmax of cosine distance between tokens (GvT Eq. 1-2;
             emphasizes dissimilar neighbors, countering over-smoothing).
        """
        l, m, _ = coords.shape
        eye = torch.eye(m, device=tok.device)
        d2 = torch.cdist(coords, coords) ** 2  # (l, m, m)
        d2_noself = d2 + eye * torch.finfo(d2.dtype).max
        knn_d2, knn_idx = d2_noself.topk(self.knn_k, dim=-1, largest=False)
        sigma2 = knn_d2.reshape(l, -1).median(dim=-1).values.clamp_min(1e-12)
        A_I = torch.exp(-d2 / (2 * sigma2[:, None, None]))
        mask = torch.zeros_like(A_I).scatter_(-1, knn_idx, 1.0)
        mask = torch.maximum(mask, mask.transpose(-2, -1))  # symmetric k-NN
        A_I = A_I * mask * (1 - eye)
        A_I = A_I.unsqueeze(0).expand(n, l, m, m).reshape(n * l, m, m)

        z = F.normalize(tok, dim=-1)
        E = F.softmax(1.0 - torch.bmm(z, z.transpose(-2, -1)), dim=-1)
        return sym_norm((A_I + eye) * E)

    def forward(self, x):
        # x: (n, l, C, p1, p2) — raw block input incl. grid channels
        n, l = x.shape[0], x.shape[1]
        V = self.freq_in_channels
        # grid planes = last 2 vars of the t=0 group; batch elem 0 (grid is static)
        gxy = x[0, :, V - 2 : V]  # (l, 2, p1, p2)
        coords = rearrange(
            gxy, "l c (h g1) (w g2) -> l (h w) c (g1 g2)", g1=self.g1, g2=self.g2
        ).mean(-1)  # (l, m, 2) physical sub-patch centroids
        tokens = rearrange(
            x, "n l c (h g1) (w g2) -> (n l) (h w) (c g1 g2)", g1=self.g1, g2=self.g2
        )
        tok = self.embed(tokens)
        A_hat = self._adjacency(tok, coords, n)
        for layer in self.layers:
            tok = layer(tok, A_hat)
        out = self.head(tok)  # (n*l, m, out_channel*g1*g2)
        h = x.shape[3] // self.g1
        out = rearrange(
            out,
            "(n l) (h w) (c g1 g2) -> n l c (h g1) (w g2)",
            n=n,
            h=h,
            c=self.out_channel,
            g1=self.g1,
            g2=self.g2,
        )
        return out


class StFTBlockG(StFTBlock):
    """StFTBlock + graph channel. graph_depth=0 disables the branch (baseline)."""

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
        patch_size=None,
        per_pixel_channels=None,
        graph_patch=(8, 8),
        graph_depth=2,
        knn_k=8,
    ):
        super(StFTBlockG, self).__init__(
            cond_time,
            freq_in_channels,
            in_dim,
            out_dim,
            out_channel,
            num_patches,
            modes,
            lift_channel=lift_channel,
            dim=dim,
            depth=depth,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            act=act,
            grid_size=grid_size,
            layer_indx=layer_indx,
        )
        if graph_depth > 0:
            self.graph = GraphBranch(
                per_pixel_channels,
                out_channel,
                freq_in_channels,
                patch_size,
                graph_patch,
                dim=dim,
                depth=graph_depth,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                act=act,
                knn_k=knn_k,
            )
        else:
            self.graph = None

    def forward(self, x):
        # identical to StFTBlock.forward, plus the graph channel summed at the end
        x_copy = x
        n, l, _, ph, pw = x.shape
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
        x_added = torch.cat((x_added, grid_dup), axis=-1)
        x = torch.cat((x_or, x_added), axis=-2)
        x = self.p(x)
        x = rearrange(x, "n l ph pw t v -> (n l) v t ph pw")
        x_ft = torch.fft.rfftn(x, dim=[2, 3, 4])[
            :, :, :, : self.modes[0], : self.modes[1]
        ]
        x_ft_real = (x_ft.real).flatten(1)
        x_ft_imag = (x_ft.imag).flatten(1)
        x_ft_real = rearrange(x_ft_real, "(n l) D -> n l D", n=n, l=l)
        x_ft_imag = rearrange(x_ft_imag, "(n l) D -> n l D", n=n, l=l)
        x_ft_real_imag = torch.cat((x_ft_real, x_ft_imag), axis=-1)
        x = self.linear(x_ft_real_imag)
        x = x + self.pos_embed_fno
        for layer in self.encoder_layers_fno:
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
        x = rearrange(x, "(n l) v t ph pw -> (n l) ph pw (v t)", n=n, l=l, t=1)
        x = self.down(x)
        x_fno = rearrange(x, "(n l) ph pw c -> n l c ph pw", n=n, l=l)
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
        if self.graph is not None:
            x = x + self.graph(x_copy)  # graph channel (true-geometry-aware)
        return x


class StFTG(StFT):
    """StFT built from StFTBlockG blocks; forward is inherited unchanged."""

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
        graph_patch_sizes=None,
        graph_depth=2,
        knn_k=8,
    ):
        nn.Module.__init__(self)

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
            per_pixel = in_channels if depth == 0 else in_channels + out_channels
            blocks.append(
                StFTBlockG(
                    cond_time,
                    num_vars,
                    p1 * p2 * per_pixel,
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
                    layer_indx=0 if depth == 0 else 1,
                    patch_size=(p1, p2),
                    per_pixel_channels=per_pixel,
                    graph_patch=graph_patch_sizes[depth],
                    graph_depth=graph_depth,
                    knn_k=knn_k,
                )
            )

        self.blocks = nn.ModuleList(blocks)
