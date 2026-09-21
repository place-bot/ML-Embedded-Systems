"""Reload the final checkpoint, verify it, and evaluate the full test set."""
import argparse
from pathlib import Path
from statistics import mean

import torch

from common import (ROOT, checkpoint_provenance, course_config, environment, evaluate_loader,
                    load_checkpoint, new_directory, run_cli, setup_cpu, write_json)
from data import check_checkpoint_data, loader, split_spec, test_set, training_sets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--verify-only', action='store_true', help='Verify saved logits and validation accuracy without accessing the test dataset.')
    args = parser.parse_args()
    setup_cpu()
    model, checkpoint, path = load_checkpoint(args.checkpoint)
    check_checkpoint_data(checkpoint)
    dataset, _, validation = training_sets(args.data_dir)
    index = checkpoint['verification']['index']
    if index != split_spec()['benchmark_index']:
        raise ValueError('Checkpoint reload probe does not match the published validation sample.')
    with torch.inference_mode():
        actual = model(dataset[index][0].unsqueeze(0))
    expected = torch.tensor(checkpoint['verification']['logits'], dtype=torch.float32)
    if not torch.allclose(actual, expected, rtol=1e-5, atol=1e-6):
        raise ValueError('Reloaded output differs from the saved verification logits.')
    validation_metrics = evaluate_loader(model, loader(validation))
    if abs(validation_metrics['accuracy_pct'] - checkpoint['verification']['validation_accuracy_pct']) > 1e-9:
        raise ValueError('Reloaded validation accuracy differs from the saved value.')
    output = new_directory(args.output)
    if args.verify_only:
        write_json(output / 'reload_verification.json', {
            **checkpoint_provenance(checkpoint, path), 'reload_verified': True,
            'validation_accuracy_pct': validation_metrics['accuracy_pct'],
            'reload_max_abs_error': (actual - expected).abs().max().item(),
            'environment': environment()})
        print(f'PASS reload: {checkpoint["model_name"]}; validation {validation_metrics["accuracy_pct"]:.2f}%. No test evaluation requested.')
        return
    test_metrics = evaluate_loader(model, loader(test_set(args.data_dir)))
    result = {**checkpoint_provenance(checkpoint, path), 'schema_version': 1,
              'test_accuracy_pct': test_metrics['accuracy_pct'], 'test_correct': test_metrics['correct'],
              'test_count': test_metrics['count'], 'test_mean_loss': test_metrics['mean_loss'],
              'validation_accuracy_pct': validation_metrics['accuracy_pct'],
              'reload_verified': True, 'reload_max_abs_error': (actual - expected).abs().max().item(),
              'mean_train_epoch_s': mean(row['train_epoch_s'] for row in checkpoint['training_history']),
              'environment': environment()}
    write_json(output / 'evaluation.json', result)
    print(f'PASS reload: {checkpoint["model_name"]}; test {test_metrics["correct"]}/10000 '
          f'= {test_metrics["accuracy_pct"]:.2f}%')


if __name__ == '__main__':
    run_cli(main)
