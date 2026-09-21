"""Inspect a completed model's tensor shapes and parameter count."""
import argparse
import torch

from common import MODEL_NAMES, parameter_count, run_cli, setup_cpu
from models import LeNet, get_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODEL_NAMES, default='baseline')
    args = parser.parse_args()
    setup_cpu()
    model = LeNet(get_config(args.model)).eval()
    inputs = torch.zeros(2, 1, 28, 28)
    with torch.inference_mode():
        features = model.features(inputs)
        logits = model(inputs)
    if tuple(logits.shape) != (2, 10):
        raise ValueError('forward must return shape [2, 10] for this two-image check.')
    print(f'input: {list(inputs.shape)}; features: {list(features.shape)}')
    print(f'flattened: [2, {features[0].numel()}]; logits: {list(logits.shape)}')
    print(f'PASS: {parameter_count(model):,} parameters; architecture {model.config}')


if __name__ == '__main__':
    run_cli(main)
