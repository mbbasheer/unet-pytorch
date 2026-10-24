"""PyTorch port of the original U-Net, matching u-net-release/phseg_v5-train.prototxt.

Differences from the textbook U-Net in the paper, all copied from the prototxt
so the released weights load and behave identically:
  * valid (unpadded) 3x3 convs, so a 572x572 input gives a 388x388 output
  * a ReLU after every up-convolution
  * the last up-convolution keeps 128 channels (not 64), so the final block
    sees 128 + 64 = 192 channels and the net has 31,100,354 parameters
  * dropout 0.5 after the d3 and d4 blocks (training only)
  * concatenation order is [upsampled, cropped skip]
"""

import math

import torch
from torch import nn

from .caffemodel import read_caffemodel

# Context lost on each side: a valid input of size 16k + 60 gives an output of 16k - 124.
BORDER = 92
N_PARAMS = 31_100_354

# Caffe layer name (from the prototxt) -> PyTorch module path.
CAFFE_TO_TORCH = {
    "conv_d0a-b": "down0.conv1",
    "conv_d0b-c": "down0.conv2",
    "conv_d1a-b": "down1.conv1",
    "conv_d1b-c": "down1.conv2",
    "conv_d2a-b": "down2.conv1",
    "conv_d2b-c": "down2.conv2",
    "conv_d3a-b": "down3.conv1",
    "conv_d3b-c": "down3.conv2",
    "conv_d4a-b": "down4.conv1",
    "conv_d4b-c": "down4.conv2",
    "upconv_d4c_u3a": "up3.upconv",
    "conv_u3b-c": "up3.convs.conv1",
    "conv_u3c-d": "up3.convs.conv2",
    "upconv_u3d_u2a": "up2.upconv",
    "conv_u2b-c": "up2.convs.conv1",
    "conv_u2c-d": "up2.convs.conv2",
    "upconv_u2d_u1a": "up1.upconv",
    "conv_u1b-c": "up1.convs.conv1",
    "conv_u1c-d": "up1.convs.conv2",
    "upconv_u1d_u0a": "up0.upconv",
    "conv_u0b-c": "up0.convs.conv1",
    "conv_u0c-d": "up0.convs.conv2",
    "conv_u0d-score": "score",
}


def valid_size(k):
    """Input and output size for the k-th valid tile size (k=32 gives 572 -> 388)."""
    return 16 * k + 60, 16 * k - 124


def center_crop(x, size):
    """Crop the last two dims of x to `size`, keeping the centre (Caffe's CROP layer)."""
    h, w = size
    top = (x.shape[-2] - h) // 2
    left = (x.shape[-1] - w) // 2
    return x[..., top:top + h, left:left + w]


class DoubleConv(nn.Module):
    """Two unpadded 3x3 conv + ReLU, optionally followed by dropout."""

    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3)
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        return self.dropout(torch.relu(self.conv2(x)))


class Up(nn.Module):
    """2x2 up-convolution + ReLU, concat with the cropped skip, then DoubleConv."""

    def __init__(self, in_channels, up_channels, skip_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, up_channels, 2, stride=2)
        self.convs = DoubleConv(up_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = torch.relu(self.upconv(x))
        skip = center_crop(skip, x.shape[-2:])
        return self.convs(torch.cat([x, skip], dim=1))


class UNetPhseg(nn.Module):
    """The phseg_v5 U-Net. Input (N, 1, H, W) with H, W of the form 16k + 60."""

    def __init__(self, in_channels=1, n_classes=2):
        super().__init__()
        # ceil_mode matches Caffe's pooling; it only matters for odd sizes.
        self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        self.down0 = DoubleConv(in_channels, 64)
        self.down1 = DoubleConv(64, 128)
        self.down2 = DoubleConv(128, 256)
        self.down3 = DoubleConv(256, 512, dropout=0.5)
        self.down4 = DoubleConv(512, 1024, dropout=0.5)
        self.up3 = Up(1024, 512, 512, 512)
        self.up2 = Up(512, 256, 256, 256)
        self.up1 = Up(256, 128, 128, 128)
        self.up0 = Up(128, 128, 64, 64)
        self.score = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        d0 = self.down0(x)
        d1 = self.down1(self.pool(d0))
        d2 = self.down2(self.pool(d1))
        d3 = self.down3(self.pool(d2))
        d4 = self.down4(self.pool(d3))
        u = self.up3(d4, d3)
        u = self.up2(u, d2)
        u = self.up1(u, d1)
        u = self.up0(u, d0)
        return self.score(u)  # raw class scores (logits)


def init_weights(model):
    """The paper's initialisation: Gaussian with std sqrt(2 / N), biases zero.

    N is the number of inputs feeding one output value: k * k * C_in for a conv.
    For the 2x2 stride-2 up-convolutions each output pixel sees one kernel
    position per input channel, so N = C_in.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.in_channels * m.kernel_size[0] * m.kernel_size[1]
        elif isinstance(m, nn.ConvTranspose2d):
            n = m.in_channels * m.kernel_size[0] * m.kernel_size[1] // (m.stride[0] * m.stride[1])
        else:
            continue
        nn.init.normal_(m.weight, 0.0, math.sqrt(2.0 / n))
        nn.init.zeros_(m.bias)
    return model


def load_caffe_weights(model, caffemodel_path):
    """Copy every parameter blob from the .caffemodel into `model`.

    Fails loudly if a layer is missing, a shape differs, or the file holds
    parameters that were not used.
    """
    layers = read_caffemodel(caffemodel_path)
    state = {}
    for caffe_name, torch_name in CAFFE_TO_TORCH.items():
        if caffe_name not in layers:
            raise KeyError(f"Layer {caffe_name!r} not found in {caffemodel_path}")
        weight, bias = layers.pop(caffe_name)
        module = model.get_submodule(torch_name)
        # Caffe conv weights are (out, in, kh, kw) and deconv weights are
        # (in, out, kh, kw): the same layouts as Conv2d and ConvTranspose2d.
        if tuple(weight.shape) != tuple(module.weight.shape):
            raise ValueError(
                f"{caffe_name}: Caffe weight {weight.shape} vs PyTorch {tuple(module.weight.shape)}"
            )
        if bias.size != module.bias.numel():
            raise ValueError(f"{caffe_name}: Caffe bias has {bias.size} values, expected {module.bias.numel()}")
        state[f"{torch_name}.weight"] = torch.from_numpy(weight.copy())
        state[f"{torch_name}.bias"] = torch.from_numpy(bias.reshape(-1).copy())

    unused = [name for name, blobs in layers.items() if blobs]
    if unused:
        raise ValueError(f"Parameter layers in the caffemodel were not loaded: {unused}")
    model.load_state_dict(state, strict=True)
    return model


def load_pretrained(caffemodel_path, device=None):
    """Build UNetPhseg, load the released weights and put it in eval mode."""
    model = load_caffe_weights(UNetPhseg(), caffemodel_path)
    if device is not None:
        model = model.to(device)
    return model.eval()
