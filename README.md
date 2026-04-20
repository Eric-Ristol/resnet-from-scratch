# ResNet-20 on CIFAR-10

A ResNet implementation for CIFAR-10 image classification, built from the paper "Deep Residual Learning for Image Recognition" (He et al., 2016).

**[Live demo](https://huggingface.co/spaces/EricRistol/resnet-cifar10)**

## The key idea

Instead of just stacking layers that try to learn y = H(x), a residual block learns the difference: F(x) = H(x) - x.

Then it outputs: y = F(x) + x

The "+ x" (skip connection) makes it easier for deep networks to train. If the optimal mapping is close to identity, the block just needs to learn something small instead of forcing a complicated function.

In the code:

```python
def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)), inplace=True)
    out = self.bn2(self.conv2(out))
    out = out + x                           # the skip connection
    return F.relu(out, inplace=True)
```

## Architecture

ResNet-20 has 20 layers (6n + 2, where n=3).

```
Input: 32x32 RGB image
  conv1 (16 filters)
  3 stages of residual blocks with stride 1, 2, 2
  global average pool
  FC layer -> 10 classes
```

Total: 269,722 parameters.

## Training

Used SGD with momentum (0.9) and weight decay (1e-4). Trained for 200 epochs on CIFAR-10.

Results: achieved ~92% test accuracy.

## How to run

```bash
python main.py
```

Menu:
- I: Generate CIFAR-10 dataset (downloads first time)
- II: Train ResNet-20
- III: Evaluate on test set
- IV: Test on a single image
- V: Quit

Or run scripts directly:
```bash
python train.py         # train and save
python evaluate.py      # test accuracy
```

## Files

```
├── model.py             ResNet-20 architecture
├── train.py             training loop
├── evaluate.py          test evaluation
├── main.py              CLI menu
├── models/              saved weights
└── plots/               training curves
```

---

**[Live demo](https://huggingface.co/spaces/EricRistol/resnet-cifar10)**
