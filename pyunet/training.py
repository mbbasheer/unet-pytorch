"""Training helpers for the ISBI 2012 EM segmentation challenge.

  * load_isbi2012     read the four .tif stacks
  * weight_maps       the paper's per-pixel loss weights (Eq. 2)
  * TileSampler       random, augmented training tiles, made on the GPU
  * weighted_cross_entropy, pixel_error, rand_score, evaluate
"""

from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from scipy import ndimage

from .tiling import predict, softmax

MEMBRANE, CELL = 0, 1  # class indices; the .tif labels store these as 0 and 255


def load_isbi2012(root):
    """Return train/test volumes as uint8 (30, 512, 512) and labels as class indices."""
    root = Path(root)

    def read(name):
        return tifffile.imread(root / f"{name}.tif")

    return {
        "train_images": read("train-volume"),
        "train_labels": (read("train-labels") > 127).astype(np.int64),
        "test_images": read("test-volume"),
        "test_labels": (read("test-labels") > 127).astype(np.int64),
    }


def class_weights(labels, n_classes=2):
    """w_c from Eq. 2: rarer classes count more, scaled so the average pixel weight is 1."""
    counts = np.bincount(np.ravel(labels), minlength=n_classes)
    return counts.sum() / (n_classes * counts)


def border_weights(label, w0=10.0, sigma=5.0):
    """The w0 * exp(-(d1 + d2)^2 / (2 sigma^2)) term of Eq. 2, on membrane pixels only.

    d1 and d2 are the distances to the nearest and second-nearest cell. Each cell's
    distance map is only computed in a window around it, because the term is
    negligible once d1 + d2 is more than about 4 sigma.
    """
    cells, n_cells = ndimage.label(label == CELL)
    d1 = np.full(label.shape, np.inf)
    d2 = np.full(label.shape, np.inf)
    margin = int(np.ceil(4 * sigma))
    for k, box in enumerate(ndimage.find_objects(cells), start=1):
        window = tuple(
            slice(max(s.start - margin, 0), min(s.stop + margin, size))
            for s, size in zip(box, label.shape)
        )
        dist = ndimage.distance_transform_edt(cells[window] != k)
        near, second = d1[window], d2[window]  # views, so writing them updates d1 and d2
        second[...] = np.where(dist < near, near, np.minimum(second, dist))
        near[...] = np.minimum(near, dist)
    border = w0 * np.exp(-((d1 + d2) ** 2) / (2 * sigma**2))
    return np.where(label == MEMBRANE, border, 0.0).astype(np.float32)


def weight_maps(labels, w0=10.0, sigma=5.0, class_w=None):
    """Full Eq. 2 weight map for each label image in a stack (N, H, W)."""
    if class_w is None:
        class_w = class_weights(labels)
    return np.stack(
        [class_w[label].astype(np.float32) + border_weights(label, w0, sigma) for label in labels]
    )


class TileSampler:
    """Draws random training tiles with the paper's augmentation, all on the GPU.

    Each call picks a random image and a random 388 x 388 output window, then
    builds the 572 x 572 input around it. Augmentation moves the sampling
    coordinates rather than the pixels: a random rotation and flip, plus a
    smooth elastic warp from random displacements on a coarse 3 x 3 grid.
    Coordinates that fall outside the image are mirrored back in, which is the
    same mirror padding used at inference time.
    """

    def __init__(self, images, labels, weights, device, in_size=572, out_size=388,
                 elastic_sigma=10.0, rotate=True, flip=True, intensity=0.1, seed=0):
        def to_gpu(a):
            return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)[:, None]

        self.images = to_gpu(images)
        self.labels = to_gpu(labels)  # float so grid_sample can read it; cast back below
        self.weights = to_gpu(weights)
        self.device = device
        self.in_size, self.out_size = in_size, out_size
        self.border = (in_size - out_size) // 2
        self.elastic_sigma, self.rotate, self.flip, self.intensity = elastic_sigma, rotate, flip, intensity
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_size=1, augment=True):
        """Return (x, y, w): images (B, 1, 572, 572), labels (B, 388, 388), weights (B, 388, 388)."""
        tiles = [self._one(augment) for _ in range(batch_size)]
        return tuple(torch.cat(parts) for parts in zip(*tiles))

    def _one(self, augment):
        rng, size, border = self.rng, self.in_size, self.border
        n, _, height, width = self.images.shape
        i = rng.integers(n)
        oy = rng.integers(0, max(height - self.out_size, 0) + 1)
        ox = rng.integers(0, max(width - self.out_size, 0) + 1)

        # Pixel coordinates (row, column) in the source image for every input pixel.
        steps = torch.arange(size, device=self.device, dtype=torch.float32)
        gy, gx = torch.meshgrid(steps + (oy - border), steps + (ox - border), indexing="ij")

        if augment:
            cy, cx = oy - border + (size - 1) / 2, ox - border + (size - 1) / 2
            dy, dx = gy - cy, gx - cx
            if self.flip and rng.random() < 0.5:
                dx = -dx
            angle = rng.uniform(0, 2 * np.pi) if self.rotate else 0.0
            cos, sin = np.cos(angle), np.sin(angle)
            gy, gx = cy + cos * dy - sin * dx, cx + sin * dy + cos * dx
            if self.elastic_sigma > 0:
                coarse = rng.normal(0.0, self.elastic_sigma, size=(1, 2, 3, 3))
                coarse = torch.as_tensor(coarse, dtype=torch.float32, device=self.device)
                field = F.interpolate(coarse, size=(size, size), mode="bicubic", align_corners=True)[0]
                gy, gx = gy + field[0], gx + field[1]

        # grid_sample wants (x, y) coordinates scaled to [-1, 1].
        grid = torch.stack([2 * gx / (width - 1) - 1, 2 * gy / (height - 1) - 1], dim=-1)[None]
        inner = grid[:, border:border + self.out_size, border:border + self.out_size]

        def sample(source, coords, mode):
            return F.grid_sample(source[i:i + 1], coords, mode=mode,
                                 padding_mode="reflection", align_corners=True)

        x = sample(self.images, grid, "bilinear")
        y = sample(self.labels, inner, "nearest")[:, 0].long()
        w = sample(self.weights, inner, "bilinear")[:, 0]

        if augment and self.intensity:
            s = self.intensity
            x = x * rng.uniform(1 - s, 1 + s) + rng.uniform(-s, s)
        return x, y, w


def weighted_cross_entropy(logits, target, weight):
    """Eq. 1 with the weight map, divided by the total weight (as the authors' Caffe layer does)."""
    per_pixel = F.cross_entropy(logits, target, reduction="none")
    return (weight * per_pixel).sum() / weight.sum()


def pixel_error(prob_cell, label, threshold=0.5):
    """Fraction of pixels whose predicted class (cell if P(cell) > threshold) is wrong."""
    return float(((prob_cell > threshold) != (label == CELL)).mean())


def segment(prob_cell, threshold=0.5):
    """Turn a probability map into numbered cells, growing each cell over the membrane pixels."""
    cells, n = ndimage.label(prob_cell > threshold)
    if n == 0:
        return cells
    _, (iy, ix) = ndimage.distance_transform_edt(cells == 0, return_indices=True)
    return cells[iy, ix]


def rand_score(pred_cells, label):
    """Foreground-restricted Rand F-score (V_rand style): 1 is perfect.

    Compares which pixels are grouped into the same cell, counting only pixels
    that are cell (not membrane) in the ground truth. It is an approximation of
    the challenge's Fiji metric, which also sweeps thresholds and thins membranes.
    """
    true_cells, _ = ndimage.label(label == CELL)
    mask = true_cells > 0
    p = pred_cells[mask].astype(np.int64)
    t = true_cells[mask].astype(np.int64)
    joint = np.bincount(p * (t.max() + 1) + t).astype(np.float64)
    sum_joint = (joint**2).sum()
    sum_pred = (np.bincount(p).astype(np.float64) ** 2).sum()
    sum_true = (np.bincount(t).astype(np.float64) ** 2).sum()
    return float(2 * sum_joint / (sum_pred + sum_true))


def evaluate(model, images, labels, flip_average=False):
    """Predict whole images and return (P(cell) maps, {metric: mean value})."""
    was_training = model.training
    model.eval()
    probs = np.stack([softmax(predict(model, img, flip_average=flip_average))[CELL] for img in images])
    model.train(was_training)
    metrics = {
        "pixel_error": float(np.mean([pixel_error(p, l) for p, l in zip(probs, labels)])),
        "rand_score": float(np.mean([rand_score(segment(p), l) for p, l in zip(probs, labels)])),
    }
    return probs, metrics
