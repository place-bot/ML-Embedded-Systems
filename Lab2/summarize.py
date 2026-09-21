"""Check provenance and combine three standard runs into the report table."""
import argparse
import csv
import math
from pathlib import Path
from statistics import mean

from benchmark import statistics
from common import (MODEL_NAMES, ROOT, course_config, new_directory, read_json, run_cli,
                    sha256, write_json)


def collect_run(run):
    run = Path(run)
    training = read_json(run / 'training_summary.json')
    evaluation = read_json(run / 'evaluation' / 'evaluation.json')
    benchmark = read_json(run / 'benchmark' / 'summary.json')
    configuration = read_json(run / 'config.json')
    digest = sha256(run / 'checkpoint.pt')
    cfg = course_config()
    for result in (training, evaluation, benchmark):
        if result['checkpoint_sha256'] != digest:
            raise ValueError(f'Mixed checkpoint results in {run}.')
        for key in ('model_name', 'architecture', 'parameter_count', 'split_sha256', 'course_config'):
            if result[key] != training[key]:
                raise ValueError(f'Inconsistent {key} across results in {run}.')
    if configuration != {'model_name': training['model_name'], 'architecture': training['architecture'], 'course_config': cfg}:
        raise ValueError(f'Configuration file differs from the measured recipe in {run}.')
    if training['course_config'] != cfg or training['split_sha256'] != sha256(ROOT / 'assets' / 'split.json'):
        raise ValueError(f'{run} does not use the published course recipe and split.')
    if evaluation['test_count'] != 10000 or not evaluation['reload_verified']:
        raise ValueError('Final results require reload verification and all 10,000 test samples.')
    accuracy = 100 * evaluation['test_correct'] / evaluation['test_count']
    if not math.isclose(accuracy, evaluation['test_accuracy_pct'], abs_tol=1e-10):
        raise ValueError('Test accuracy does not match the recorded correct count.')
    if (benchmark['threads'], benchmark['warmup'], benchmark['runs'], benchmark['input_shape']) != (1, 10, 100, [1, 1, 28, 28]):
        raise ValueError('The required table uses one thread, 10 warm-ups, 100 calls and batch size one.')
    with (run / 'benchmark' / 'raw_timings.csv').open(newline='', encoding='utf-8') as stream:
        raw = list(csv.DictReader(stream))
    if [int(row['iteration']) for row in raw] != list(range(1, 101)):
        raise ValueError('Expected exactly 100 numbered timing rows.')
    recalculated = statistics([float(row['latency_ms']) for row in raw])
    for key in ('median_ms', 'p95_ms', 'mean_ms'):
        if not math.isclose(recalculated[key], benchmark['inference'][key], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f'{key} differs from the raw timing CSV.')
    with (run / 'training.csv').open(newline='', encoding='utf-8') as stream:
        history = list(csv.DictReader(stream))
    if [int(row['epoch']) for row in history] != list(range(1, cfg['epochs'] + 1)):
        raise ValueError('Training CSV has an unexpected epoch count.')
    train_mean = mean(float(row['train_epoch_s']) for row in history)
    if not math.isfinite(train_mean) or train_mean <= 0:
        raise ValueError('Training times must be finite and positive.')
    for result in (training, evaluation):
        if not math.isclose(train_mean, result['mean_train_epoch_s'], rel_tol=1e-12):
            raise ValueError('Training mean differs from the per-epoch CSV.')
    return {'model_name': training['model_name'], 'architecture': training['architecture'],
            'parameter_count': training['parameter_count'], 'test_accuracy_pct': accuracy,
            'mean_train_epoch_s': train_mean, 'inference_median_ms': recalculated['median_ms'],
            'inference_p95_ms': recalculated['p95_ms'], 'checkpoint_sha256': digest,
            'input_sha256': benchmark['input_sha256'], 'benchmark_environment': benchmark['environment']}


def collect_runs(runs):
    rows = [collect_run(run) for run in runs]
    if sorted(row['model_name'] for row in rows) != sorted(MODEL_NAMES):
        raise ValueError('Supply one baseline, one conv_small and one fc_small run.')
    if len({row['input_sha256'] for row in rows}) != 1:
        raise ValueError('All three benchmarks must use the same input tensor.')
    if len({(row['benchmark_environment']['pi_model'], row['benchmark_environment']['torch'],
             row['benchmark_environment']['machine']) for row in rows}) != 1:
        raise ValueError('The comparison must use the same device model and runtime version.')
    rows.sort(key=lambda row: MODEL_NAMES.index(row['model_name']))
    for row in rows:
        row['speedup_vs_baseline'] = rows[0]['inference_median_ms'] / row['inference_median_ms']
        row['accuracy_change_pp'] = row['test_accuracy_pct'] - rows[0]['test_accuracy_pct']
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs=3, type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = collect_runs(args.runs)
    output = new_directory(args.output)
    write_json(output / 'comparison.json', {'schema_version': 1, 'models': rows})
    columns = ('model_name', 'parameter_count', 'test_accuracy_pct', 'mean_train_epoch_s',
               'inference_median_ms', 'speedup_vs_baseline', 'accuracy_change_pp')
    with (output / 'comparison.csv').open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f'{row["model_name"]:12} {row["parameter_count"]:6} parameters; '
              f'{row["test_accuracy_pct"]:.2f}%; {row["mean_train_epoch_s"]:.2f} s/epoch; '
              f'{row["inference_median_ms"]:.4f} ms')


if __name__ == '__main__':
    run_cli(main)
