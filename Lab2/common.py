"""Shared CPU setup, provenance, checkpoint loading and result-file helpers."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
MODEL_NAMES = ('baseline', 'conv_small', 'fc_small')
LABELS = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')


def read_json(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream)


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def course_config():
    return read_json(ROOT / 'course_config.json')


def setup_cpu(threads=1, seed=42):
    if type(threads) is not int or threads < 1:
        raise ValueError('Thread count must be a positive integer.')
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def new_directory(path):
    destination = Path(path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f'{destination} already exists. Use a new --output directory.')
    destination.mkdir(parents=True)
    return destination


def device_snapshot():
    def read_text(path):
        try:
            return Path(path).read_text().strip('\0\n ')
        except OSError:
            return None
    temperature = read_text('/sys/class/thermal/thermal_zone0/temp')
    frequency = read_text('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq')
    throttled = None
    try:
        result = subprocess.run(['vcgencmd', 'get_throttled'], capture_output=True,
                                text=True, timeout=3, check=True)
        throttled = result.stdout.strip().split('=', 1)[-1]
    except (OSError, subprocess.SubprocessError):
        pass
    return {'temperature_c': float(temperature) / 1000 if temperature else None,
            'cpu_frequency_khz': int(frequency) if frequency else None,
            'throttled_hex': throttled}


def environment():
    try:
        pi_model = Path('/proc/device-tree/model').read_text().strip('\0\n ')
    except OSError:
        pi_model = None
    return {'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'python': platform.python_version(), 'system': platform.platform(),
            'machine': platform.machine(), 'pi_model': pi_model,
            'torch': str(torch.__version__), 'numpy': str(np.__version__),
            'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
            'execution_device': 'cpu'}


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def load_checkpoint(path):
    from models import LeNet, validate_config
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    try:
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as exc:
        raise ValueError(f'Cannot read checkpoint {path.name}: {exc}') from exc
    required = {'schema_version', 'model_name', 'architecture', 'state_dict',
                'course_config', 'split_sha256', 'data_manifest_sha256', 'parameter_count',
                'training_history', 'verification'}
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError('Checkpoint is missing required Lab 2 metadata.')
    if checkpoint['schema_version'] != 1 or checkpoint['model_name'] not in MODEL_NAMES:
        raise ValueError('Unsupported checkpoint version or model name.')
    config = validate_config(checkpoint['architecture'], checkpoint['model_name'])
    if checkpoint['course_config']['preprocessing'] != course_config()['preprocessing']:
        raise ValueError('Checkpoint preprocessing does not match the course input contract.')
    model = LeNet(config).cpu()
    try:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    except RuntimeError as exc:
        raise ValueError(f'Checkpoint weights do not match its architecture: {exc}') from exc
    if parameter_count(model) != checkpoint['parameter_count']:
        raise ValueError('Checkpoint parameter count does not match its architecture.')
    if not all(torch.isfinite(value).all().item() for value in model.state_dict().values()):
        raise ValueError('Checkpoint contains non-finite weights.')
    model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 1, 28, 28))
    if tuple(output.shape) != (1, 10):
        raise ValueError('Model must return ten logits for each image.')
    return model, checkpoint, path


def checkpoint_provenance(checkpoint, path):
    return {'model_name': checkpoint['model_name'], 'architecture': checkpoint['architecture'],
            'parameter_count': checkpoint['parameter_count'], 'checkpoint_sha256': sha256(path),
            'split_sha256': checkpoint['split_sha256'],
            'data_manifest_sha256': checkpoint['data_manifest_sha256'],
            'course_config': checkpoint['course_config']}


def evaluate_loader(model, loader):
    model.eval()
    correct, count, loss_sum = 0, 0, 0.0
    with torch.inference_mode():
        for inputs, targets in loader:
            logits = model(inputs)
            loss_sum += torch.nn.functional.cross_entropy(logits, targets, reduction='sum').item()
            correct += (logits.argmax(dim=1) == targets).sum().item()
            count += targets.numel()
    if count == 0 or not math.isfinite(loss_sum):
        raise ValueError('Evaluation needs nonempty data and finite logits.')
    return {'correct': correct, 'count': count, 'accuracy_pct': 100.0 * correct / count,
            'mean_loss': loss_sum / count}


def run_cli(main):
    try:
        main()
    except (ValueError, FileNotFoundError, FileExistsError, NotImplementedError, RuntimeError,
            OSError, KeyError) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        sys.exit(2)
