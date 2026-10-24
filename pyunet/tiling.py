"""Overlap-tile inference, ported from u-net-release/mycaffe_tiled_forward5.m.

The image is mirror-padded by BORDER pixels, cut into tiles whose outputs sit
side by side, and each tile is run through the net (optionally averaged over
the four flips, as the MATLAB code does).

Axis note: the MATLAB code transposes images, so its single tiled axis is the
image's rows. Here everything is in normal (H, W) order and n_tiles=(2, 1)
reproduces the original U373 setup: two tiles stacked vertically, each with a
444 x 892 input and a 260 x 708 output.
"""

import math

import numpy as np
import torch

from .model import BORDER, valid_size

_FLIPS = [(), (-1,), (-2,), (-2, -1)]


def to_float(image):
    """Scale an image to [0, 1] like MATLAB's im2double (uint8 / 255, uint16 / 65535)."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.integer):
        return image.astype(np.float32) / np.iinfo(image.dtype).max
    return image.astype(np.float32)


def tile_geometry(shape, n_tiles):
    """Return (input_size, output_size) per axis for splitting `shape` into n_tiles."""
    in_size, out_size = [], []
    for size, n in zip(shape, n_tiles):
        # Smallest k whose output, 16k - 124, covers this tile's share of the image.
        k = max(8, math.ceil((math.ceil(size / n) + 124) / 16))
        i, o = valid_size(k)
        in_size.append(i)
        out_size.append(o)
    return tuple(in_size), tuple(out_size)


@torch.inference_mode()
def _forward_tile(model, tile, flip_average):
    flips = _FLIPS if flip_average else _FLIPS[:1]
    total = 0
    for dims in flips:
        x = tile.flip(dims) if dims else tile
        y = model(x)
        total = total + (y.flip(dims) if dims else y)
    return total / len(flips)


def predict(model, image, n_tiles=(1, 1), flip_average=True, device=None):
    """Segment a 2-D image (H, W) or multi-channel image (C, H, W).

    Returns the raw class scores as a float32 array (n_classes, H, W). Use
    `scores.argmax(0)` for labels (1 = cell) or `softmax(scores)` for probabilities.
    """
    image = to_float(image)
    if image.ndim == 2:
        image = image[None]
    _, height, width = image.shape
    device = device or next(model.parameters()).device

    (in_h, in_w), (out_h, out_w) = tile_geometry((height, width), n_tiles)
    n_y, n_x = n_tiles

    # Mirror the border (excluding the edge pixel, as the MATLAB code does), then
    # zero-fill so the last tile fits. The zeros only feed outputs past the image.
    padded = np.pad(image, ((0, 0), (BORDER, BORDER), (BORDER, BORDER)), mode="reflect")
    extra_h = n_y * out_h + 2 * BORDER - padded.shape[1]
    extra_w = n_x * out_w + 2 * BORDER - padded.shape[2]
    padded = np.pad(padded, ((0, 0), (0, max(extra_h, 0)), (0, max(extra_w, 0))))
    padded = torch.from_numpy(padded)

    scores = None
    for ty in range(n_y):
        for tx in range(n_x):
            y0, x0 = ty * out_h, tx * out_w
            tile = padded[None, :, y0:y0 + in_h, x0:x0 + in_w].to(device)
            out = _forward_tile(model, tile, flip_average)[0].float().cpu().numpy()
            if scores is None:
                scores = np.zeros((out.shape[0], n_y * out_h, n_x * out_w), dtype=np.float32)
            scores[:, y0:y0 + out_h, x0:x0 + out_w] = out
    return scores[:, :height, :width]


def softmax(scores, axis=0):
    """Per-pixel class probabilities from raw scores."""
    e = np.exp(scores - scores.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)
