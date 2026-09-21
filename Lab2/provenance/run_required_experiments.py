"""Run the instructor commands sequentially on the Pi; leave their code unchanged."""
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path('/home/jingyi/CSE60685-FA26-Lab-2')
sys.path.insert(0, str(ROOT))
from common import device_snapshot

PYTHON = str(ROOT / 'env/bin/python')
SETUP = ROOT / 'results/setup'
CONTEXT = SETUP / 'run_context.json'
record = {
    'device': 'Supplied Raspberry Pi 4B; all required training, evaluation and timings run on its CPU.',
    'power': 'Supplied CanaKit USB-C adapter.',
    'cooling': 'Case lid closed; fan running, confirmed by student on 2026-09-21; held unchanged.',
    'governor': Path('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor').read_text().strip(),
    'execution': 'Sequential; unchanged instructor train.py, evaluate.py, benchmark.py, summarize.py.',
    'pre_launch_temperature_limit_c': 40.0,
    'stages': [],
}


def now():
    return datetime.now(timezone.utc).isoformat()


def save():
    CONTEXT.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')


def cool():
    deadline = time.monotonic() + 600
    while True:
        state = device_snapshot()
        if state['throttled_hex'] != '0x0':
            raise RuntimeError(f'Power/throttling flags need inspection: {state}')
        if state['temperature_c'] is not None and state['temperature_c'] <= 40.0:
            print('Ready at', state, flush=True)
            return state
        if time.monotonic() >= deadline:
            raise RuntimeError(f'Cooling target not reached; preserve existing results: {state}')
        print('Waiting for comparable starting temperature:', state, flush=True)
        time.sleep(15)


def run(label, arguments, cool_first=False):
    before = cool() if cool_first else device_snapshot()
    item = {'label': label, 'command': [PYTHON, *arguments],
            'started_utc': now(), 'before': before}
    record['stages'].append(item)
    save()
    print('\nSTART', label, item['started_utc'], flush=True)
    with (SETUP / (label + '.log')).open('x', encoding='utf-8') as log:
        proc = subprocess.Popen([PYTHON, '-u', *arguments], cwd=ROOT,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        code = proc.wait()
    item.update(finished_utc=now(), exit_code=code, after=device_snapshot())
    save()
    if code:
        raise RuntimeError(f'{label} failed with exit code {code}; retain files and inspect.')
    print('PASS', label, flush=True)


def main():
    if CONTEXT.exists():
        raise FileExistsError('Run context already exists; do not overwrite a previous experiment.')
    for name in ('baseline', 'conv_small', 'fc_small'):
        if (ROOT / 'results' / name).exists():
            raise FileExistsError(f'Existing {name} results must be preserved.')
    subprocess.run(['git', 'diff', '--exit-code', '--', 'train.py', 'benchmark.py',
                    'evaluate.py', 'summarize.py', 'common.py', 'data.py', 'course_config.json'],
                   cwd=ROOT, check=True)
    run('train_baseline', ['train.py', '--model', 'baseline', '--output', 'results/baseline'], True)
    run('baseline_reload', ['evaluate.py', '--checkpoint', 'results/baseline/checkpoint.pt',
                           '--verify-only', '--output', 'results/baseline/reload_check'])
    for name in ('conv_small', 'fc_small'):
        run('train_' + name, ['train.py', '--model', name, '--output', f'results/{name}'], True)
    record['all_training_completed_utc'] = now()
    save()
    for name in ('baseline', 'conv_small', 'fc_small'):
        run('evaluate_' + name, ['evaluate.py', '--checkpoint', f'results/{name}/checkpoint.pt',
                                '--output', f'results/{name}/evaluation'])
    for name in ('baseline', 'conv_small', 'fc_small'):
        run('benchmark_' + name, ['benchmark.py', '--checkpoint', f'results/{name}/checkpoint.pt',
                                 '--threads', '1', '--warmup', '10', '--runs', '100',
                                 '--output', f'results/{name}/benchmark'], True)
    run('comparison', ['summarize.py', '--runs', 'results/baseline', 'results/conv_small',
                      'results/fc_small', '--output', 'results/comparison'])
    record['required_experiments_completed_utc'] = now()
    save()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if record['stages']:
            record['error'] = str(error)
            save()
        raise
