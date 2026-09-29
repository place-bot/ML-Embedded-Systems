"""Audit exported Lab 3 evidence without executing models on the OMEN."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit(root):
    results = root / 'results'
    context = read(results / 'setup/run_context.json')
    checks = []

    def check(condition, description):
        if not condition:
            raise AssertionError(description)
        checks.append(description)

    check(bool(context.get('required_experiments_completed_utc')) and not context.get('error'),
          'Required execution completed without a recorded error')
    expected_names = ['import_baseline', 'check_pruning', 'prune_models']
    names = ['control', 's25', 's50']
    expected_names += [prefix + name for prefix in ['finetune_', 'evaluate_', 'benchmark_'] for name in names]
    expected_names += ['comparison']
    stages = context['stages']
    check([s['label'] for s in stages] == expected_names, 'Correct import, pruning, training, evaluation, timing order')
    check(context['source']['sha256'] == '47071f5d25f7aa5fef7414a187e6ea0ddf9ef12e01de022ff453a3d5ca9b870d',
          'Own Lab 2 A checkpoint identity')
    check(context['instructor_commit'] == '4c635083c39862381940db06d134bfe95ce92088', 'Reviewed instructor revision')
    check(context['instructor_file_sha256'] == context['instructor_file_sha256_after'], 'Instructor files unchanged through execution')
    for name, value in context['instructor_file_sha256'].items():
        check(digest(root / name) == value, f'Unmodified instructor file: {name}')
    for index, stage in enumerate(stages):
        check(stage['exit_code'] == 0, f"Successful command: {stage['label']}")
        if index:
            check(datetime.fromisoformat(stages[index-1]['finished_utc']) <= datetime.fromisoformat(stage['started_utc']),
                  f"Sequential execution: {stage['label']}")
        if 'input_sha256' in stage:
            check(stage['input_sha256'] == stage['input_sha256_after'], f"Immutable input: {stage['label']}")
            if stage['label'] != 'import_baseline':
                relative = stage['input_checkpoint'].split('/CSE60685-FA26-Lab-3/', 1)[1]
                check(digest(root / relative) == stage['input_sha256'], f"Retained input checkpoint: {stage['label']}")
        for name, value in stage.get('output_sha256', {}).items():
            check(digest(root / name) == value, f"Retained exact stage output: {name}")
    source = read(results / 'source/source.json')
    check(source['origin']['used_fallback'] is False and source['origin']['lab2_epochs'] == 5,
          'Personal five-epoch baseline imported')
    check(source['parameter_count'] == 44426 and source['validation']['accuracy_pct'] == 79.65,
          'Source parameter count and saved validation reproduced')
    rows = read(results / 'comparison/comparison.json')
    check([r['model'] for r in rows] == names, 'Three required comparison rows')
    cfg = read(root / 'course_config.json')
    signature = None
    for row, c2, parameters in zip(rows, [16, 12, 8], [44426, 36142, 27858]):
        name = row['model']
        folder = results / name
        config = read(folder / 'config.json')
        train = read(folder / 'training_summary.json')
        evaluation = read(folder / 'test/evaluation.json')
        timing = read(folder / 'timing/summary.json')
        pruned = read(results / 'pruned' / name / 'pruning.json')
        check(config['architecture'] == dict(c1=6, c2=c2, h1=120, h2=84), f'{name}: fixed architecture')
        check(config['course_config'] == cfg, f'{name}: fixed training settings')
        check(config['origin'] == source['origin'] == pruned['origin'], f'{name}: common source origin')
        check(row['parameters'] == train['parameter_count'] == pruned['parameter_count'] == parameters,
              f'{name}: correct parameter count')
        if name != 'control':
            keep = config['pruning']['keep_indices']
            check(len(keep) == c2 and keep == sorted(set(keep)) and set(keep) <= set(range(16)), f'{name}: retained channels')
            check(config['pruning'] == pruned['pruning'], f'{name}: pruning metadata preserved during fine-tuning')
        with (folder / 'training.csv').open(newline='') as stream:
            history = list(csv.DictReader(stream))
        check([int(r['epoch']) for r in history] == [1,2,3], f'{name}: three complete fine-tuning epochs')
        check(float(history[-1]['validation_accuracy_pct']) == train['validation_after_ft_pct'], f'{name}: last-epoch result')
        check(pruned['before_validation']['accuracy_pct'] == train['validation_before_ft_pct'] == row['validation_before_ft_pct'],
              f'{name}: immediate pre-fine-tuning validation retained')
        check(evaluation['reload_verified'] and evaluation['test_count'] == 10000, f'{name}: reload verification and full test set')
        check(evaluation['test_accuracy_pct'] == evaluation['test_correct'] / 100 == row['test_accuracy_pct'], f'{name}: test accuracy arithmetic')
        check(evaluation['validation_accuracy_pct'] == train['validation_after_ft_pct'] == row['validation_after_ft_pct'],
              f'{name}: validation agreement')
        for data in [train, evaluation, timing]:
            env = data['environment']
            check(env['machine'] == 'aarch64' and 'Raspberry Pi 4' in env['pi_model'] and env['execution_device'] == 'cpu',
                  f'{name}: actual Pi CPU execution in {data.get("runtime", "training/evaluation")}')
            check(env['threads'] == env['interop_threads'] == 1, f'{name}: one intra-op and one inter-op thread')
        check(timing['threads'] == 1 and timing['warmup'] == 10 and timing['runs'] == 100, f'{name}: correct benchmark recipe')
        current = (timing['input_index'], timing['input_label'], timing['input_shape'], timing['input_dtype'])
        if signature is None:
            signature = current
        check(current == signature and current[2:] == ([1,1,28,28], 'float32'), f'{name}: same prepared image')
        with (folder / 'timing/raw_timings.csv').open(newline='') as stream:
            samples = list(csv.DictReader(stream))
        values = [float(r['latency_ms']) for r in samples]
        check([int(r['iteration']) for r in samples] == list(range(1,101)), f'{name}: exactly 100 timing records')
        check(all(math.isfinite(v) and v > 0 for v in values), f'{name}: finite positive timings')
        check(math.isclose(statistics.median(values), timing['inference']['median_ms'], abs_tol=1e-12), f'{name}: independently recomputed median')
        check(row['median_ms'] == timing['inference']['median_ms'], f'{name}: summary median agrees')
        check(math.isclose(row['speedup_vs_control'], rows[0]['median_ms']/row['median_ms']), f'{name}: speedup arithmetic')
        check(all(s['throttled_hex'] == '0x0' for s in timing['telemetry'].values()), f'{name}: no sampled power/throttling flags')
    return {'status': 'PASS', 'check_count': len(checks), 'checks': checks, 'comparison': rows,
            'scope': 'Read-only exported file, metadata, provenance and arithmetic checks; no model execution on OMEN.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f"PASS: {result['check_count']} evidence checks. No models ran on OMEN.")
