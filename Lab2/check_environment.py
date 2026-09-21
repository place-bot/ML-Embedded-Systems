"""Check dependencies and CPU execution without importing exercises."""
import importlib.metadata
import platform

import torch
from torch import nn
import torchvision

from common import ROOT, course_config, environment, read_json, run_cli, setup_cpu


def main():
    setup_cpu()
    print(f'Python {platform.python_version()}; machine {platform.machine()}')
    for package in ('torch', 'torchvision', 'numpy', 'Pillow'):
        print(f'{package}: {importlib.metadata.version(package)}')
    for filename in ('data_manifest.json', 'split.json'):
        read_json(ROOT / 'assets' / filename)
    model = nn.Sequential(nn.Conv2d(1, 2, 5), nn.ReLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2, 10))
    logits = model(torch.zeros(2, 1, 28, 28))
    nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert logits.shape == (2, 10) and all(torch.isfinite(p.grad).all() for p in model.parameters())
    print('PASS: a CPU forward and backward pass completed; course assets are readable.')
    if platform.machine() not in ('aarch64', 'arm64'):
        print('Development host detected. Required latency measurements must be collected on the assigned Pi.')
    print('Next: python prepare_data.py, then complete the Tutorial exercises.')


if __name__ == '__main__':
    run_cli(main)
