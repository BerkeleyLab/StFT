import torch
from StFT_3D import StFT
from torchinfo import summary

cond_time = 1
num_vars = 5  # physical_vars (1) + grid coords (2)
patch_sizes = [(128, 128), (64, 64), (32, 32)]
overlaps = [(1, 1), (1, 1), (1, 1)]
in_channels = cond_time * num_vars
out_channels = num_vars - 2
modes = [(8, 8), (8, 8), (8, 8)]
img_size = (256, 256)
lift_channel = 64
dim = 512
vit_depth = [6, 6, 6]
num_heads = 1
mlp_dim = 512

model = StFT(
    cond_time=cond_time,
    num_vars=num_vars,
    patch_sizes=patch_sizes,
    overlaps=overlaps,
    in_channels=in_channels,
    out_channels=out_channels,
    modes=modes,
    img_size=img_size,
    lift_channel=lift_channel,
    dim=dim,
    vit_depth=vit_depth,
    num_heads=num_heads,
    mlp_dim=mlp_dim,
)

batch_size = 2
x = torch.randn(batch_size, cond_time, num_vars - 2, *img_size)
grid = torch.randn(1, 2, *img_size)

print(model)
print("\n" + "=" * 80 + "\n")

summary(model, input_data=(x, grid), depth=4, col_names=["input_size", "output_size", "num_params", "mult_adds"])

