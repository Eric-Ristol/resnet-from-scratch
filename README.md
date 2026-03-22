# ResNet-20 on CIFAR-10 — from scratch

A clean, single-folder reimplementation of the CIFAR-10 ResNet from
**Deep Residual Learning for Image Recognition** (He, Zhang, Ren & Sun, CVPR 2016
— [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)).

The goal of this project isn't to beat state-of-the-art — it's to go from the
equations in the paper to working code, one layer at a time, with nothing
imported from `torchvision.models`.

---

## 1. The idea in one equation

A plain deep network stacks layers that each try to learn some mapping
`y = H(x)` directly. As networks get deeper, optimisation gets harder —
gradients shrink, training accuracy actually goes *down* even though the
model has more capacity.

A residual block instead learns the **residual** `F(x) = H(x) - x`, and outputs

```
    y = F(x, {W_i}) + x
```

The `+ x` is the *identity shortcut*. If the optimal mapping H is close to
identity (empirically very common for nearby layers), the block only needs to
push F toward zero — much easier than forcing a nonlinear stack of conv+BN+ReLU
to reproduce the input.

In `model.py` this lives inside `BasicBlock.forward`:

```python
def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)), inplace=True)
    out = self.bn2(self.conv2(out))
    out = out + self._shortcut(x)         # ← y = F(x) + x
    return F.relu(out, inplace=True)
```

The `_shortcut` helper handles the two cases from the paper:

- **Identity shortcut** — when `in_channels == out_channels` and `stride == 1`,
  the shortcut is literally just `x`.
- **Option A projection-free shortcut** — when the block halves the spatial
  resolution and doubles the channels, we downsample `x` with a stride-2
  avg-pool and zero-pad the extra channels. No learnable parameters.
  This is the variant He et al. used for their CIFAR experiments.

---

## 2. Architecture (Table 6 of the paper → code)

The CIFAR variants are defined by a single integer `n`. Total depth is
`6n + 2` layers (the `+ 2` accounts for the first conv and the final FC).

| layer name | output size | ResNet-20 (n=3)                          |
|------------|-------------|------------------------------------------|
| conv1      | 32×32×16    | 3×3, 16 filters, stride 1                |
| conv2_x    | 32×32×16    | 3 × BasicBlock(16)                       |
| conv3_x    | 16×16×32    | 3 × BasicBlock(32) — first block stride 2|
| conv4_x    | 8×8×64      | 3 × BasicBlock(64) — first block stride 2|
| avgpool    | 1×1×64      | global average pool                      |
| fc         | 10          | fully-connected → 10 classes             |

In `model.py` that's exactly:

```python
self.conv1  = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
self.bn1    = nn.BatchNorm2d(16)
self.stage1 = self._make_stage(in_ch=16, out_ch=16, blocks=n, stride=1)
self.stage2 = self._make_stage(in_ch=16, out_ch=32, blocks=n, stride=2)
self.stage3 = self._make_stage(in_ch=32, out_ch=64, blocks=n, stride=2)
self.avgpool = nn.AdaptiveAvgPool2d(1)
self.fc      = nn.Linear(64, num_classes)
```

`count_parameters(resnet20())` returns **269,722**, which matches the ~0.27 M
figure reported in the paper.

**Why no bias on the convs?** The following BatchNorm has its own learnable
shift, so a bias on the conv is redundant. The paper drops it; so do we.

**Why He (Kaiming) init?** Initialises conv weights so that the variance
of activations is preserved through ReLU layers — critical for training
deep nets from scratch without warm-up. In `_init_weights`:

```python
nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
```

---

## 3. Training recipe

The paper's exact recipe (Sec. 4.2):

- SGD, momentum 0.9, weight decay 1e-4
- mini-batch 128
- learning rate 0.1, divided by 10 at epochs 80 and 120, trained for 160 epochs
- random crop with 4-pixel zero-padding + random horizontal flip augmentation
- He initialisation

This reimplementation defaults to:

- 80 epochs with a **cosine annealing** LR schedule
- Nesterov momentum (a tiny improvement over vanilla momentum, "free")
- Same batch size, weight decay, data augmentation, and init

Why the change? Cosine annealing gets you within ~1% of the step-decay recipe
in roughly half the epochs and produces a smoother learning curve — nicer for
a README plot and faster to iterate on.

In `train.py`:

```python
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                      weight_decay=1e-4, nesterov=True)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
```

---

## 4. Expected results

| Model        | Params | Test error (He et al., 2016) | Our target (80-epoch cosine) |
|--------------|--------|------------------------------|------------------------------|
| ResNet-20    | 0.27 M | 8.75 %                       | ~9 %                         |
| ResNet-32    | 0.46 M | 7.51 %                       | —                            |

"Our target" is the accuracy you should expect on a single modern GPU run. If
you see test accuracy plateau much below ~91 %, something is off — the most
common culprits are (a) forgetting to normalise, (b) a broken shortcut path,
or (c) a scheduler that's already at ~0 LR when meaningful training starts.

---

## 5. How to run

Assumes the shared venv in `../venv-all/` (created for all projects in
`Projects/`).

```bash
# from Projects/ResNet-from-scratch/
source ../venv-all/bin/activate
pip install -r requirements.txt    # first time only

python main.py
```

`main.py` shows a Roman-numeral menu:

```
I.   Download CIFAR-10
II.  Train ResNet-20 (80 epochs)
III. Smoke-train (tiny subset, 1 epoch)
IV.  Evaluate best checkpoint
V.   Predict on a single image
VI.  Launch API server (web demo)
VII. Quit
```

Or use the scripts directly:

```bash
python train.py --epochs 80            # full training
python train.py --tiny --epochs 1      # ~10s smoke test
python evaluate.py                     # reports per-class accuracy + confusion matrix
pytest                                 # ~5s, verifies model shapes & invariants
```

Artifacts land in:

- `models/best.pt` — best checkpoint (by test accuracy)
- `models/history.json` — per-epoch loss / accuracy / LR
- `plots/learning_curves.png` — loss + accuracy over epochs
- `plots/confusion_matrix.png` — row-normalised 10×10 class confusion

---

## 6. Files

```
model.py        — BasicBlock + ResNetCifar + resnet20() / resnet32() factories
data.py         — CIFAR-10 download, augmentation, DataLoaders
train.py        — training loop, cosine LR, checkpointing, history plot
evaluate.py     — best-checkpoint eval + per-class accuracy + confusion matrix
main.py         — Roman-numeral menu (I - VII)
test_model.py   — pytest invariants (shapes, param counts, BN behaviour)
api/
  app.py        — FastAPI server (image upload → classification)
  static/
    index.html  — drag-and-drop web demo
```

---

## 7. Web demo (API)

After training, launch the API server:

```bash
python main.py          # pick option VI
# or directly:
uvicorn api.app:app --reload
```

Then open **http://localhost:8000** in your browser. Drag and drop any image (or click to browse), hit Classify, and see the top-5 predictions with confidence bars. The model resizes any image to 32x32 internally, so any resolution works.

Endpoints:
- `GET /` — the drag-and-drop web demo
- `POST /predict` — upload an image file, get back label + confidence + top-5 predictions
- `GET /health` — server health check

---

## 8. What I'd do next

The model here is intentionally the paper's original baseline. Things that
are cheap to try and would teach something:

- **Swap option-A shortcut for option-B (1×1 conv projection).** Costs a tiny
  number of parameters but consistently improves test accuracy by a few
  tenths of a percent. A good way to see that a parameter-free design isn't
  automatically better.
- **Add MixUp or CutMix augmentation.** Both typically give +1–2 % on
  CIFAR-10 with zero architectural change. Clean isolation of "regularisation
  vs. capacity" as two separate knobs.
- **Train the ResNet-56 variant (n=9).** The paper's interesting claim is
  that depth keeps helping up to ~110 layers on CIFAR. Verifying that
  empirically is the point of the whole paper.
- **Profile where training time actually goes.** The CIFAR ResNet is small
  enough that DataLoader throughput, not the GPU, is usually the bottleneck
  on a laptop — a good teaching moment about where compute really lives in
  modern training.
- **Try stochastic depth (Huang et al., 2016).** A minimal change to
  `BasicBlock.forward` and a small but consistent test-accuracy win.

---

## 9. Gotchas I hit while building this

Writing these down so future-me doesn't repeat them:

- **Inplace ReLU after the addition is fine here.** But not if you ever
  branch off `out` again below the sum — the in-place op would corrupt
  the tensor on the other branch. Easy bug to miss in a more complex block.
- **Option-A shortcut needs padding on the *channel* dim, not spatial.**
  `F.pad` with the 6-tuple `(0,0,0,0,0,pad)` — easy to get the order wrong.
- **BN `train()` vs `eval()` matters a lot** for reproducibility checks.
  `test_eval_mode_is_deterministic` in `test_model.py` catches the case
  where you forget to `.eval()` the model before comparing forward passes.
- **`drop_last=False` on the test loader.** With 10,000 test images and
  batch size 256 the final batch has 16 images — dropping it silently
  biases your reported accuracy.
