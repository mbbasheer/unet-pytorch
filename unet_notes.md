# U-Net: Study Notes

Paper: *U-Net: Convolutional Networks for Biomedical Image Segmentation*, Ronneberger, Fischer, Brox (2015), arXiv:1505.04597

Written for someone new to neural nets and segmentation, with the maths and the plain-words version side by side.

---

## Contents

1. [Paper summary](#1-paper-summary)
2. [Core vocabulary](#2-core-vocabulary)
3. [Filters and channels](#3-filters-and-channels)
4. [Architecture walkthrough](#4-architecture-walkthrough)
5. [Why the output is smaller, and the overlap-tile strategy](#5-why-the-output-is-smaller-and-the-overlap-tile-strategy)
6. [What the output is](#6-what-the-output-is)
7. [Loss function (the "energy")](#7-loss-function-the-energy)
8. [Training: initialisation, SGD, momentum, batch size](#8-training-initialisation-sgd-momentum-batch-size)
9. [Data augmentation](#9-data-augmentation)
10. [Why these design choices](#10-why-these-design-choices)
11. [Parameter count](#11-parameter-count)
12. [Experiments and results](#12-experiments-and-results)
13. [Implementation tips](#13-implementation-tips)
14. [Errata and gotchas](#14-errata-and-gotchas)

---

## 1. Paper summary

**Problem.** Biomedical image segmentation means assigning a class label to *every pixel* (for example cell vs background). Annotated biomedical data is scarce (tens of images, not thousands), and the output needs precise localisation.

**Prior approach (Ciresan et al., 2012).** A sliding-window network predicts each pixel's class from a local patch around it. It won ISBI 2012 but has two drawbacks:
- It is slow: the network runs once per patch, with lots of redundant computation on overlapping patches.
- There is a trade-off between localisation accuracy and context. Big patches need more pooling, which hurts localisation; small patches see little context.

**Idea.** Build on the Fully Convolutional Network (Long et al. 2015). Use a contracting path to capture context and a symmetric expanding path to recover resolution, with skip connections that copy high-resolution features across. The shape is a "U".

**Key ingredients.**
1. U-shaped encoder-decoder with skip connections (copy-and-crop) and many feature channels in the upsampling path.
2. Valid (unpadded) convolutions plus an overlap-tile strategy with mirror padding, so arbitrarily large images can be segmented seamlessly.
3. Heavy elastic-deformation data augmentation, so very few annotated images suffice.
4. A pixel-wise weighted cross-entropy loss that forces the network to learn the thin background gaps between touching cells.

**Results.**
- Best warping error on the ISBI 2012 EM neuronal-structure challenge at the time (0.000353 vs 0.000420 for Ciresan et al.).
- Won the ISBI Cell Tracking Challenge 2015 by a large margin on two transmitted-light datasets: PhC-U373 (IOU 92% vs 83%) and DIC-HeLa (IOU 77.5% vs 46%).
- Fast: a 512x512 image takes under a second on a recent GPU. Training took about 10 hours on an NVIDIA Titan (6 GB).

---

## 2. Core vocabulary

| Term | Meaning |
|---|---|
| Segmentation | Assigning a class to every pixel |
| Feature map | A stack of 2D grids (channels) produced by a layer |
| Channel | One 2D grid in that stack |
| Filter / kernel | A small grid of learnable weights that produces one output channel |
| Valid / unpadded conv | No padding: border positions where the filter would overhang are skipped, so the output shrinks |
| ReLU | `max(0, x)`, applied elementwise |
| Max pooling | Take the maximum in each non-overlapping 2x2 block, halving width and height |
| Up-convolution | Learned 2x upsampling (doubles width and height) |
| Skip connection | Copying an earlier feature map across to a later layer |
| Softmax | Turns raw scores into probabilities |
| Epoch | One pass through the training data |
| Batch | Number of images processed before one weight update |
| Hyperparameter | A setting you choose (learning rate, momentum, w0, sigma), not learned |

---

## 3. Filters and channels

### Channel

A **channel** is one 2D grid of numbers.
- A grayscale image has 1 channel; RGB has 3.
- A "64-channel feature map" is 64 grids stacked, all with the same height and width.
- Each grid answers one question across the whole image, such as "how vertical-edge-like is it here?"

### Filter

A **filter** is a small grid of learnable numbers that slides across the input to produce **one** output channel.

For a single-channel input `x`, 3x3 filter `w`, bias `b`:

```
y(i,j) = b + sum_{u=0..2} sum_{v=0..2} w(u,v) * x(i+u, j+v)
```

In words: lay the 3x3 filter over a 3x3 patch, multiply matching numbers, add them all up, add the bias. Slide to the next position and repeat.

**Example.** `w = [[-1,0,1],[-1,0,1],[-1,0,1]]`
- On a vertical-edge patch `[[0,0,9],[0,0,9],[0,0,9]]` the output is 27 (lights up).
- On a flat patch of all 5s the output is 0.
- So this filter detects vertical edges.

### Multiple input channels

If the input has `C_in` channels, the filter is 3x3x`C_in` and also sums across channels:

```
y(i,j) = b + sum_c sum_u sum_v w(c,u,v) * x_c(i+u, j+v)
```

### A layer with many filters

A conv layer with `C_out` filters has `C_out` independent (w, b) sets, giving `C_out` output channels. Then ReLU: `a = max(0, y)`.

Parameters in one conv layer: `(k * k * C_in + 1) * C_out`
- Layer 1 (1 to 64 channels): 64 x (9 + 1) = 640
- Layer 2 (64 to 64 channels): 64 x (9*64 + 1) = 36,928

The same filter weights are reused at every position (**weight sharing**), so an edge detector finds edges anywhere in the image.

### Answering "how does 572x572x1 become 570x570x64?"

- 64 different filters each slide over the 572x572 input, giving 64 output grids.
- Each output grid is 570x570 because a 3x3 window cannot be centred on the outermost pixel ring: 572 - 2 = 570.
- The second conv takes the 64 channels in, uses filters of size 3x3x64, and gives 64 channels out at 568x568.

---

## 4. Architecture walkthrough

### Contracting path (left side, "what is in the image")

Repeated blocks of:
1. 3x3 conv (unpadded) + ReLU
2. 3x3 conv (unpadded) + ReLU
3. 2x2 max pool, stride 2 (halves height and width)

Channels double at each level: 64, 128, 256, 512, 1024.

### Expanding path (right side, "where it is")

Repeated blocks of:
1. 2x2 up-convolution (doubles height and width, halves channels)
2. Concatenate with the **cropped** feature map from the matching contracting level (the skip connection)
3. 3x3 conv + ReLU
4. 3x3 conv + ReLU

### Final layer

A 1x1 conv maps each pixel's 64-component vector to K class scores (K = 2 in the paper).

Total: **23 convolutional layers**, no fully connected layers.

### Shape trace (paper's Fig. 1, 572x572 input)

| Stage | Operation | Size x channels |
|---|---|---|
| Input | | 572x572 x 1 |
| Down 1 | conv, conv | 570 x 64, then 568x568 x 64 (saved for skip) |
| | pool | 284x284 x 64 |
| Down 2 | conv, conv | 282 x 128, then 280x280 x 128 (skip) |
| | pool | 140x140 x 128 |
| Down 3 | conv, conv | 138 x 256, then 136x136 x 256 (skip) |
| | pool | 68x68 x 256 |
| Down 4 | conv, conv | 66 x 512, then 64x64 x 512 (skip) |
| | pool | 32x32 x 512 |
| Bottom | conv, conv | 30 x 1024, then 28x28 x 1024 |
| Up 4 | up-conv | 56x56 x 512 |
| | concat with cropped 64x64 skip (crop to 56) | 56x56 x 1024 |
| | conv, conv | 54 x 512, then 52x52 x 512 |
| Up 3 | up-conv, concat (crop 136 to 104), conv, conv | 104 x 512 -> 102 -> 100 x 256 |
| Up 2 | up-conv, concat (crop 280 to 200), conv, conv | 200 x 256 -> 198 -> 196 x 128 |
| Up 1 | up-conv, concat (crop 568 to 392), conv, conv | 392 x 128 -> 390 -> 388 x 64 |
| Output | 1x1 conv | **388x388 x 2** |

Why the two paths exist:
- **Down path:** pooling shrinks the image so each later filter "sees" a big region (context), but precise position is lost.
- **Up path:** restores resolution, but on its own it only knows coarse location.
- **Skip connections (grey arrows):** carry high-resolution early features straight across. The decoder then combines "what" (deep, contextual) with "where" (shallow, precise).
- The skip maps must be **cropped** because they are larger than the upsampled maps (unpadded convs lost borders along the way).

**Tile size constraint:** every 2x2 max pool must act on a map with even height and width, so the input tile size has to be chosen carefully (572 works; arbitrary sizes may not).

---

## 5. Why the output is smaller, and the overlap-tile strategy

Each valid conv loses 1 pixel per side. Through the whole network the output ends up 388x388 for a 572x572 input: a border of 92 pixels is lost on each side. So each output pixel is predicted using a context window extending about 92 px beyond it, and only pixels whose full context lies inside the input are output.

**Overlap-tile strategy (Fig. 2):**
- To segment a big image, cut it into tiles.
- For each tile, feed in a larger window (blue area) and get out the smaller central region (yellow area).
- Neighbouring input windows overlap, but the output regions tile seamlessly with no gaps.
- At the image edges, the missing context is **extrapolated by mirroring** the image.
- This keeps memory use bounded so arbitrarily large images can be handled.

---

## 6. What the output is

- The final 1x1 conv outputs **K raw scores per pixel** (one per class). These are arbitrary real numbers.
- A per-pixel **softmax** converts them to probabilities: values in [0, 1] summing to 1 over classes. Example: 0.93 cell / 0.07 background.
- For a final mask, take the argmax (or threshold the foreground probability, for example at 0.5).
- **Training labels:** the ground-truth mask, where each pixel holds its true class (for example 1 = cell, 0 = background). The ground-truth is cropped to match the 388x388 output region.
- For the EM challenge, the probability map itself is submitted and the organisers threshold it at 10 levels.

---

## 7. Loss function (the "energy")

"Energy function" is the paper's name for the **loss function**: the single number E that training minimises. "Energy", "loss", "cost" and "objective" are interchangeable here.

### Step A: pixel-wise softmax

```
p_k(x) = exp(a_k(x)) / sum_{k'=1..K} exp(a_k'(x))
```

`a_k(x)` is the raw score for class `k` at pixel `x`. Worked example: scores (2.0, -1.0) give exp (7.39, 0.37), total 7.76, so p = (0.95, 0.05).

### Step B: weighted cross-entropy

```
E = - sum_{x in Omega} w(x) * log( p_{l(x)}(x) )
```

- `l(x)` is the true class of pixel `x`; only the probability given to the **true** class matters.
- `w(x)` is a per-pixel weight map.

| p given to the true class | -log p | Meaning |
|---|---|---|
| 0.99 | 0.01 | confident and right: tiny penalty |
| 0.5 | 0.69 | unsure |
| 0.1 | 2.30 | mostly wrong |
| 0.01 | 4.61 | confidently wrong: huge penalty |

Nice property of softmax + cross-entropy: the gradient w.r.t. the raw score is simply

```
dE/da_k = p_k - y_k      (prediction minus one-hot truth)
```

### The weight map (Eq. 2)

```
w(x) = w_c(x) + w_0 * exp( - (d1(x) + d2(x))^2 / (2 * sigma^2) )
```

- `w_c(x)`: balances class frequencies (so the dominant class does not swamp the loss).
- `d1(x)`: distance from pixel `x` to the border of the **nearest** cell.
- `d2(x)`: distance to the border of the **second-nearest** cell.
- `w_0 = 10`, `sigma ~ 5 px` (hyperparameters, not learned).
- For a pixel in the thin gap between two touching cells, `d1 + d2` is small so the exp term is close to `w_0`, making those pixels about 10x more important. This forces the network to carve out thin separating borders (Fig. 3d).
- Pre-computed once per ground-truth mask using morphological operations.

---

## 8. Training: initialisation, SGD, momentum, batch size

### Initialisation

- **Not zeros.** If all filters in a layer start identical, they get identical gradients and stay identical forever (symmetry problem).
- **Not N(0,1).** Unit-variance weights make activations explode across ~20 layers.
- **Paper's choice:** Gaussian, mean 0, standard deviation `sqrt(2/N)`, where `N` = number of incoming connections per neuron = `3*3*C_in`. For `C_in = 64`, `N = 576`, std ~ 0.059. The factor 2 compensates for ReLU zeroing half the values, keeping each feature map at approximately unit variance (He et al. 2015).
- Biases are usually started at 0.

### How it learns (gradient descent)

1. **Forward pass:** image through the network, giving probabilities `p`.
2. **Loss:** compare with ground truth, giving the scalar `E`.
3. **Backpropagation:** chain rule gives `dE/dw` for every weight.
4. **Update:** move against the gradient.

```
w <- w - eta * dE/dw          (eta = learning rate)
```

For a conv weight, since one weight is used at every position its gradient sums over positions:

```
dE/dw(u,v) = sum_{i,j} delta(i,j) * x(i+u, j+v),     delta(i,j) = dE/dy(i,j)
```

The loss surface is non-convex, so SGD finds a good solution, not a provably optimal one. In practice that is enough.

### Batch size = 1 and large tiles

- A **batch** is the number of images processed per weight update. GPU memory is roughly (memory per tile) x (batch size).
- With valid convs a large fraction of a small tile is wasted border, so large tiles are more efficient.
- The authors therefore use **one big tile per batch** instead of several small ones.
- Cost: one image gives a noisy gradient. Momentum compensates.

### Momentum (0.99)

```
v <- mu * v - eta * grad(E)
w <- w + v                      (mu = 0.99)
```

- Intuition: a heavy ball rolling downhill. It does not change direction at every bump; noisy sideways kicks average out while a consistent direction builds up speed.
- Unrolled: `v_t = -eta * sum_k mu^k * grad(E_{t-k})`. Past gradients are weighted by `0.99^k`.
- Effective memory is about `1/(1-mu) = 100` steps, so each update is effectively an average over roughly the last 100 images. This mimics a batch of about 100 without the memory cost. (The common default of 0.9 averages over about 10 steps.)
- Side effect: in a steady direction the effective step is `eta/(1-mu) = 100*eta`, so the learning rate must be small.

---

## 9. Data augmentation

With so few annotated images, augmentation is essential. The paper needs shift and rotation invariance, robustness to deformation, and robustness to gray-value changes.

**Random elastic deformation** is the key idea:
- Sample random displacement vectors on a coarse **3x3 grid**, from a Gaussian with **10 px** standard deviation.
- Interpolate to per-pixel displacements with **bicubic interpolation**, giving smooth warps.
- Applied to the image and its mask together.
- Why it suits biomedical images: deformation is the most common variation in tissue, and realistic deformations can be simulated cheaply.

**Dropout** at the end of the contracting path provides further implicit augmentation.

At test time on the EM data, predictions were averaged over 7 rotated versions of the input.

---

## 10. Why these design choices

There is no derivation; these are choices guided by intuition and validated experimentally.

| Piece | Purpose |
|---|---|
| 3x3 conv + ReLU | Stacked small filters see a large area (two 3x3s see 5x5) with fewer weights than one big filter. ReLU adds non-linearity; without it stacked convs collapse into one linear op. Two convs per level follows VGG. |
| Max pool 2x2 | Halves resolution so later filters cover a bigger region of the original image; adds slight shift tolerance; saves compute. |
| Doubling channels per pool | Area drops 4x, so channels double to keep per-layer compute roughly constant and give deeper layers more capacity for abstract features. |
| Up-conv 2x2 | Learned upsampling: each coarse pixel expands into a 2x2 block via learned weights. |
| Copy-and-crop skip | Pooling discards precise position; copying early high-res maps gives the decoder "where" to combine with "what". The paper's main addition over FCN. |
| Many channels in the up path | Lets context propagate to the higher-resolution layers; makes the network symmetric (U shape). |
| Final 1x1 conv | A per-pixel linear classifier: 64 features to K class scores. |
| No fully connected layers | Lets the network accept any input size and output a map. |

Max pooling and copy-and-crop have **no parameters**; nothing is learned there.

---

## 11. Parameter count

Conv layer: `(k*k*C_in + 1) * C_out`. Up-conv (2x2): `(2*2*C_in + 1) * C_out`.

| Stage | Layers | Params |
|---|---|---|
| Down 1 | 1->64, 64->64 | 37,568 |
| Down 2 | 64->128, 128->128 | 221,440 |
| Down 3 | 128->256, 256->256 | 885,248 |
| Down 4 | 256->512, 512->512 | 3,539,968 |
| Bottom | 512->1024, 1024->1024 | 14,157,824 |
| Up 4 | up-conv 1024->512; 1024->512; 512->512 | 9,176,576 |
| Up 3 | up-conv 512->256; 512->256; 256->256 | 2,294,528 |
| Up 2 | up-conv 256->128; 256->128; 128->128 | 573,824 |
| Up 1 | up-conv 128->64; 128->64; 64->64 | 143,552 |
| Output | 1x1, 64->2 | 130 |
| **Total** | | **31,030,658 (~31.0 M)** |

Observations:
- Over 45% of the parameters sit in the two bottom convs, because cost scales with `C_in x C_out`.
- Concatenation doubles `C_in` in the first up-path conv of each level (for example 512 upsampled + 512 skipped = 1024 in), so up-path layers are larger than their down-path mirrors.
- 31 M parameters from about 30 images is why augmentation matters so much.
- With `padding=1` convs the parameter count is identical.

**Not learned (hyperparameters):** learning rate, momentum (0.99), `w_0` (10), `sigma` (~5), elastic deformation settings, tile size.

---

## 12. Experiments and results

### EM segmentation (ISBI 2012)
- Data: 30 training images, 512x512, serial-section TEM of Drosophila larva ventral nerve cord, with ground truth for cells (white) and membranes (black). Test labels are secret; you submit a probability map.
- Metrics: warping error, Rand error, pixel error (thresholding at 10 levels).

| Rank | Group | Warping error | Rand error | Pixel error |
|---|---|---|---|---|
| 1 | u-net | 0.000353 | 0.0382 | 0.0611 |
| 2 | DIVE-SCI | 0.000355 | 0.0305 | 0.0584 |
| 3 | IDSIA (Ciresan) | 0.000420 | 0.0504 | 0.0613 |

### ISBI Cell Tracking Challenge 2015 (IOU)

| Method | PhC-U373 | DIC-HeLa |
|---|---|---|
| Second best 2015 | 0.83 | 0.46 |
| **u-net** | **0.9203** | **0.7756** |

- PhC-U373: glioblastoma-astrocytoma cells, phase contrast, 35 partially annotated training images.
- DIC-HeLa: HeLa cells on glass, DIC microscopy, 20 partially annotated training images.
- IOU = intersection over union = |prediction AND truth| / |prediction OR truth|.

---

## 13. Implementation tips

- **Start with `padding=1`.** Most modern implementations use same-padding so output size equals input size. It avoids all cropping logic and is much easier to start with. Switch to valid convs later if you want to follow the paper exactly.
- **Skip the weight map at first.** Plain cross-entropy (or Dice loss) is enough to get something working; add Eq. 2 later.
- **Framework:** PyTorch is the easiest route.
- **Loss:** `nn.CrossEntropyLoss()` applies the softmax internally, so feed it raw scores, not probabilities. Targets are integer class indices of shape (N, H, W).
- **Optimiser:** `torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99)` (small LR because of the 100x effective step). Adam is a common modern alternative.
- **Init:** `nn.init.kaiming_normal_(w, nonlinearity='relu')` matches the paper's `sqrt(2/N)`.
- **Augmentation:** use elastic deformation (for example the `albumentations` library's `ElasticTransform`), plus flips/rotations/intensity jitter. Apply identical geometric transforms to image and mask; use nearest-neighbour interpolation for the mask.
- **Sanity checks:** print the parameter count (should be ~31.03 M for 2 classes with the original channel widths); overfit a single image first to check the training loop; check the input size is compatible with 4 poolings (divisible by 16 with same-padding).
- **Data:** the ISBI 2012 EM challenge data (30 training images, 512x512) and the Cell Tracking Challenge datasets are the ones used in the paper.

---

## 14. Errata and gotchas

- **Missing minus sign in Eq. 1.** The paper prints `E = sum w(x) log p_l(x)(x)`. The correct loss to minimise is `E = - sum w(x) log p_l(x)(x)`.
- **Valid vs same padding:** the paper's network shrinks the map (572 -> 388); common re-implementations do not.
- **Tile size matters:** with valid convs, input sizes must make every pooling layer see even dimensions.
- **Softmax and CrossEntropyLoss:** do not apply softmax yourself before PyTorch's `CrossEntropyLoss`; it would be applied twice.
- **Momentum 0.99 needs a small learning rate** and batch size 1; if you use larger batches, 0.9 is a more typical choice.
- **"Energy" = loss.** Different word, same thing.
