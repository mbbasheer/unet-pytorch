"""Sanity checks for the PyTorch port of the released U-Net.

Run with:  python check_port.py
"""

import time

import numpy as np
import torch

from pyunet import (
    N_PARAMS,
    UNetPhseg,
    load_pretrained,
    predict,
    read_caffemodel,
    tile_geometry,
)
from pyunet.model import BORDER, CAFFE_TO_TORCH

CAFFEMODEL = "u-net-release/phseg_v5.caffemodel"


def check(name, ok, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        raise SystemExit(1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"torch {torch.__version__}, device: {device}\n")

    t = time.time()
    layers = read_caffemodel(CAFFEMODEL)
    with_params = {name: blobs for name, blobs in layers.items() if blobs}
    check(
        "caffemodel has exactly the 23 expected parameter layers",
        set(with_params) == set(CAFFE_TO_TORCH),
        f"read in {time.time() - t:.1f}s",
    )
    values = np.concatenate([b.ravel() for blobs in with_params.values() for b in blobs])
    check("all weights finite and not all zero", np.isfinite(values).all() and np.abs(values).max() > 0)

    model = load_pretrained(CAFFEMODEL, device)
    n = sum(p.numel() for p in model.parameters())
    check("parameter count matches the prototxt", n == N_PARAMS == values.size, f"{n:,}")

    with torch.inference_mode():
        out = model(torch.rand(1, 1, 572, 572, device=device))
    check("572x572 input gives 388x388 output (paper Fig. 1)", tuple(out.shape) == (1, 2, 388, 388))

    geom = tile_geometry((520, 696), (2, 1))
    check(
        "U373 tiling matches tmp-test.prototxt (444 x 892 input)",
        geom == ((444, 892), (260, 708)),
        str(geom),
    )

    # One tile, no flips: predict() must equal a direct forward pass on the mirrored image.
    rng = np.random.default_rng(0)
    image = rng.random((300, 200), dtype=np.float32)
    scores = predict(model, image, n_tiles=(1, 1), flip_average=False)
    (in_h, in_w), _ = tile_geometry(image.shape, (1, 1))
    padded = np.pad(image, BORDER, mode="reflect")
    padded = np.pad(padded, ((0, in_h - padded.shape[0]), (0, in_w - padded.shape[1])))
    with torch.inference_mode():
        direct = model(torch.from_numpy(padded)[None, None].to(device))[0].cpu().numpy()
    direct = direct[:, : image.shape[0], : image.shape[1]]
    diff = np.abs(scores - direct).max()
    check("tiled predict() places outputs correctly", diff < 1e-4, f"max diff {diff:.2e}")

    # Several tiles vs one: close but not bit-identical, because tile offsets
    # (16k - 124) are not multiples of 16, so the pooling grid shifts slightly.
    multi = predict(model, image, n_tiles=(2, 2), flip_average=False)
    agree = (multi.argmax(0) == scores.argmax(0)).mean()
    print(f"[INFO] 2x2 tiles vs 1 tile on noise: labels agree on {agree:.2%} of pixels")

    # The model is fresh from Caffe; make sure an untrained net differs (guards against a no-op load).
    fresh = UNetPhseg().to(device).eval()
    with torch.inference_mode():
        x = torch.rand(1, 1, 572, 572, device=device)
        changed = not torch.allclose(fresh(x), model(x))
    check("loaded weights differ from a random init", changed)

    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
