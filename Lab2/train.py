"""Train one architecture from scratch under the common course recipe."""
import argparse
import csv
import math
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch

from common import (ROOT, MODEL_NAMES, course_config, device_snapshot, environment,
                    evaluate_loader, new_directory, parameter_count, run_cli,
                    setup_cpu, sha256, write_json)
from data import loader, split_spec, training_sets
from models import LeNet, get_config
from train_step import train_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=MODEL_NAMES, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    cfg = course_config()
    setup_cpu(cfg['threads'], cfg['seed'])
    architecture = get_config(args.model)
    dataset, training, validation = training_sets(args.data_dir)
    model = LeNet(architecture).cpu()
    if model(torch.zeros(2, 1, 28, 28)).shape != (2, 10):
        raise ValueError('forward must preserve the batch dimension and return ten logits.')
    output = new_directory(args.output)
    write_json(output / 'config.json', {'model_name': args.model, 'architecture': architecture,
                                     'course_config': cfg})
    train_loader, validation_loader = loader(training, True), loader(validation)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['learning_rate'])
    loss_fn = torch.nn.CrossEntropyLoss()
    initial_weights = {name: value.clone() for name, value in model.state_dict().items()}
    before = device_snapshot()
    history = []
    print(f'{args.model}: {parameter_count(model):,} parameters, {cfg["epochs"]} epochs, CPU / one thread', flush=True)
    fields = ('epoch', 'train_loss', 'train_accuracy_pct', 'validation_loss',
              'validation_accuracy_pct', 'train_epoch_s')
    with (output / 'training.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, cfg['epochs'] + 1):
            model.train()
            loss_sum, correct, count = 0.0, 0, 0
            # TIMING START: data iteration, forward/backward/update and fixed batch statistics.
            start = perf_counter()
            for inputs, targets in train_loader:
                loss, batch_correct = train_step(model, inputs, targets, loss_fn, optimizer)
                if not math.isfinite(loss):
                    raise ValueError('Training loss became non-finite. Check your training step.')
                loss_sum += loss * targets.numel()
                correct += batch_correct
                count += targets.numel()
            elapsed = perf_counter() - start
            # TIMING END: validation, printing, CSV and checkpoint writes are outside.
            measured = evaluate_loader(model, validation_loader)
            row = {'epoch': epoch, 'train_loss': loss_sum / count,
                   'train_accuracy_pct': 100.0 * correct / count,
                   'validation_loss': measured['mean_loss'],
                   'validation_accuracy_pct': measured['accuracy_pct'], 'train_epoch_s': elapsed}
            history.append(row)
            writer.writerow(row)
            stream.flush()
            print(f'Epoch {epoch}/{cfg["epochs"]}: loss={row["train_loss"]:.4f}, '
                  f'validation={row["validation_accuracy_pct"]:.2f}%, train={elapsed:.2f} s', flush=True)
    if all(torch.equal(initial_weights[name], value) for name, value in model.state_dict().items()):
        raise ValueError('No model weights changed. Check backward() and optimizer.step().')
    model.eval()
    probe_index = split_spec()['benchmark_index']
    probe = dataset[probe_index][0].unsqueeze(0)
    with torch.inference_mode():
        probe_logits = model(probe).tolist()
    checkpoint = {'schema_version': 1, 'model_name': args.model, 'architecture': architecture,
                  'state_dict': model.state_dict(), 'course_config': cfg,
                  'split_sha256': sha256(ROOT / 'assets' / 'split.json'),
                  'data_manifest_sha256': sha256(ROOT / 'assets' / 'data_manifest.json'),
                  'parameter_count': parameter_count(model), 'training_history': history,
                  'verification': {'index': probe_index, 'logits': probe_logits,
                                   'validation_accuracy_pct': history[-1]['validation_accuracy_pct']},
                  'environment': environment()}
    checkpoint_path = output / 'checkpoint.pt'
    with checkpoint_path.open('xb') as stream:
        torch.save(checkpoint, stream)
    write_json(output / 'training_summary.json', {
        'schema_version': 1, 'model_name': args.model, 'architecture': architecture,
        'parameter_count': parameter_count(model), 'course_config': cfg,
        'mean_train_epoch_s': mean(row['train_epoch_s'] for row in history),
        'checkpoint_sha256': sha256(checkpoint_path), 'split_sha256': checkpoint['split_sha256'],
        'environment': environment(), 'telemetry': {'before': before, 'after': device_snapshot()},
        'timing_scope': 'Training loop including data iteration and fixed batch statistics; excludes validation, test, printing and saving.'})
    print(f'Saved {checkpoint_path}. Next run evaluate.py in a new process after all three architectures are fixed.')


if __name__ == '__main__':
    run_cli(main)
