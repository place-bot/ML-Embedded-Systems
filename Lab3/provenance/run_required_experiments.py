"""Execute unchanged instructor commands sequentially on the actual Pi.

This external helper records provenance and survives SSH disconnections when
launched with nohup. It does not implement or change any course model.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/home/jingyi/CSE60685-FA26-Lab-3')
sys.path.insert(0, str(ROOT))
from common import device_snapshot, environment

PYTHON = str(ROOT / 'env/bin/python')
SETUP = ROOT / 'results/setup'
CONTEXT = SETUP / 'run_context.json'
BASELINE = ROOT.parent / 'CSE60685-FA26-Lab-2/results/baseline/checkpoint.pt'
BASELINE_HASH = '47071f5d25f7aa5fef7414a187e6ea0ddf9ef12e01de022ff453a3d5ca9b870d'
UPSTREAM_COMMIT = '4c635083c39862381940db06d134bfe95ce92088'
NAMES = ('control', 's25', 's50')
record = {
    'execution_device': 'Supplied Raspberry Pi 4B CPU, not the OMEN control computer.',
    'power': 'Supplied CanaKit USB-C adapter.',
    'cooling': 'Case lid closed and fan running; confirmed by student on 2026-09-29.',
    'execution': 'Sequential execution of unchanged instructor programs.',
    'pre_benchmark_launch_temperature_limit_c': 40.0,
    'source': {'path': str(BASELINE), 'sha256': BASELINE_HASH, 'used_fallback': False},
    'stages': [],
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def save():
    temporary = CONTEXT.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    temporary.replace(CONTEXT)


def code_hashes():
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).split(b'\0')
    return {name.decode(): sha256(ROOT / name.decode()) for name in names if name}


def cool():
    deadline = time.monotonic() + 600
    while True:
        snapshot = device_snapshot()
        if snapshot['throttled_hex'] != '0x0':
            raise RuntimeError(f'Inspect power/throttling flags before measuring: {snapshot}')
        if snapshot['temperature_c'] is not None and snapshot['temperature_c'] <= 40:
            return snapshot
        if time.monotonic() >= deadline:
            raise RuntimeError(f'Cooling target not reached; retain current results: {snapshot}')
        print('Waiting to cool:', snapshot, flush=True)
        time.sleep(15)


def run(label, arguments, checkpoint=None, cool_first=False):
    if code_hashes() != record['instructor_file_sha256']:
        raise RuntimeError('Instructor files changed; stop before executing another stage.')
    before = cool() if cool_first else device_snapshot()
    item = {'label': label, 'command': [PYTHON, *arguments],
            'started_utc': now(), 'before': before}
    if checkpoint is not None:
        item['input_checkpoint'] = str(checkpoint)
        item['input_sha256'] = sha256(checkpoint)
    record['stages'].append(item)
    save()
    print('START', label, item['started_utc'], flush=True)
    with (SETUP / f'{label}.log').open('x', encoding='utf-8') as log:
        process = subprocess.Popen([PYTHON, '-u', *arguments], cwd=ROOT,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    item.update(finished_utc=now(), exit_code=code, after=device_snapshot())
    if checkpoint is not None:
        item['input_sha256_after'] = sha256(checkpoint)
    if '--output' in arguments:
        output = ROOT / arguments[arguments.index('--output') + 1]
        if output.is_dir():
            item['output_sha256'] = {
                str(path.relative_to(ROOT)): sha256(path)
                for path in sorted(output.rglob('*')) if path.is_file()
            }
    save()
    if code:
        raise RuntimeError(f'{label} failed (exit {code}); inspect preserved output.')
    if checkpoint is not None and item['input_sha256_after'] != item['input_sha256']:
        raise RuntimeError(f'{label} changed its input checkpoint.')
    print('PASS', label, flush=True)


def main():
    if CONTEXT.exists():
        raise FileExistsError('Existing experiment context: inspect it, do not overwrite.')
    for name in ('source', 'pruned', *NAMES):
        if (ROOT / 'results' / name).exists():
            raise FileExistsError(f'Existing {name} results must be preserved.')
    if sha256(BASELINE) != BASELINE_HASH:
        raise ValueError('Pi baseline differs from the verified own Lab 2 A checkpoint.')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if head != UPSTREAM_COMMIT:
        raise ValueError('Instructor revision differs from the reviewed source.')
    subprocess.run(['git', 'diff', '--exit-code', 'HEAD'], cwd=ROOT, check=True)
    record.update(started_utc=now(), instructor_commit=head,
                  instructor_file_sha256=code_hashes(), controller_environment=environment(),
                  governor=Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor').read_text().strip())
    save()
    run('import_baseline', ['import_baseline.py', '--checkpoint', str(BASELINE),
                           '--output', 'results/source'], checkpoint=BASELINE)
    run('check_pruning', ['check_pruning.py'])
    source = ROOT / 'results/source/checkpoint.pt'
    run('prune_models', ['prune_models.py', '--baseline', 'results/source/checkpoint.pt',
                         '--output', 'results/pruned'], checkpoint=source)
    for name in NAMES:
        checkpoint = ROOT / f'results/pruned/{name}/checkpoint.pt'
        run('finetune_' + name, ['finetune.py', '--checkpoint', str(checkpoint),
                                '--output', f'results/{name}'], checkpoint=checkpoint)
    record['all_finetuning_completed_utc'] = now()
    save()
    for name in NAMES:
        checkpoint = ROOT / f'results/{name}/checkpoint.pt'
        run('evaluate_' + name, ['evaluate.py', '--checkpoint', str(checkpoint),
                                '--output', f'results/{name}/test'], checkpoint=checkpoint)
    record['all_evaluation_completed_utc'] = now()
    save()
    for name in NAMES:
        checkpoint = ROOT / f'results/{name}/checkpoint.pt'
        run('benchmark_' + name, ['benchmark.py', '--checkpoint', str(checkpoint),
                                 '--threads', '1', '--warmup', '10', '--runs', '100',
                                 '--output', f'results/{name}/timing'],
            checkpoint=checkpoint, cool_first=True)
    run('comparison', ['summarize.py', '--results', 'results', '--output', 'results/comparison'])
    record['required_experiments_completed_utc'] = now()
    record['instructor_file_sha256_after'] = code_hashes()
    save()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if record['stages']:
            record['error'] = str(error)
            save()
        raise
