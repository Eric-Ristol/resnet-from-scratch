#Pytest checks for the ResNet-20 implementation.
#
#These run in a fraction of a second (no dataset, no training) and catch
#the most common "I changed the model and forgot something" bugs:
#  - output shape mismatch
#  - parameter count drift
#  - wrong dtype on the forward pass
#  - BatchNorm not respecting train/eval mode
#  - identity shortcut still working when channels change

import pytest
import torch

from model import resnet20, resnet32, BasicBlock, count_parameters


def test_resnet20_output_shape():
    m = resnet20()
    x = torch.randn(4, 3, 32, 32)
    y = m(x)
    assert y.shape == (4, 10), f"expected (4, 10), got {tuple(y.shape)}"


def test_resnet20_param_count():
    #Paper says ~0.27M params for ResNet-20 on CIFAR-10.
    n = count_parameters(resnet20())
    #Option-A shortcut has no params, so this should be tight.
    assert 260_000 < n < 280_000, f"unexpected param count {n}"


def test_resnet32_is_larger():
    assert count_parameters(resnet32()) > count_parameters(resnet20())


def test_forward_dtype_is_float32():
    m = resnet20()
    y = m(torch.randn(1, 3, 32, 32))
    assert y.dtype == torch.float32


def test_basic_block_shortcut_with_downsample():
    #When stride=2 and channels grow, the shortcut must still add cleanly.
    blk = BasicBlock(in_channels=16, out_channels=32, stride=2)
    x = torch.randn(2, 16, 16, 16)
    y = blk(x)
    assert y.shape == (2, 32, 8, 8), f"expected (2,32,8,8), got {tuple(y.shape)}"


def test_basic_block_shortcut_identity():
    #Same channels, stride 1 → shortcut is a pure identity.
    blk = BasicBlock(in_channels=16, out_channels=16, stride=1)
    x = torch.randn(2, 16, 8, 8)
    y = blk(x)
    assert y.shape == x.shape


def test_eval_mode_is_deterministic():
    #In eval mode, two forward passes on the same input should match.
    m = resnet20().eval()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        a = m(x)
        b = m(x)
    assert torch.allclose(a, b), "eval-mode forward pass is not deterministic"
