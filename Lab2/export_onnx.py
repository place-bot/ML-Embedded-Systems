"""Optional: export one trained checkpoint and verify ONNX Runtime CPU logits."""
import argparse
from pathlib import Path

import numpy as np
import torch

from common import (ROOT, checkpoint_provenance, environment, load_checkpoint,
                    new_directory, run_cli, setup_cpu, sha256, write_json)
from data import check_checkpoint_data, split_spec, training_sets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise ValueError('Install requirements-optional.txt before running this optional exercise.') from exc
    setup_cpu()
    model, checkpoint, path = load_checkpoint(args.checkpoint)
    check_checkpoint_data(checkpoint)
    dataset, _, _ = training_sets(args.data_dir)
    output = new_directory(args.output)
    exported = output / 'model.onnx'
    dummy = torch.zeros(1, 1, 28, 28)
    torch.onnx.export(model, dummy, str(exported), input_names=['image'], output_names=['logits'],
                      opset_version=17, dynamo=False, do_constant_folding=True)
    onnx.checker.check_model(onnx.load(str(exported)))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(exported), sess_options=options, providers=['CPUExecutionProvider'])
    generator = torch.Generator().manual_seed(42)
    inputs = [('zero', dummy), ('seeded_random', torch.rand((1, 1, 28, 28), generator=generator))]
    inputs += [(f'validation_{i}', dataset[i][0].unsqueeze(0)) for i in split_spec()['validation_indices'][:3]]
    checks = []
    with torch.inference_mode():
        for name, tensor in inputs:
            expected = model(tensor).numpy()
            actual = session.run(['logits'], {'image': tensor.numpy()})[0]
            if not np.allclose(actual, expected, rtol=1e-3, atol=1e-4):
                raise ValueError(f'ONNX output comparison failed on {name}.')
            checks.append({'input': name, 'max_abs_error': float(np.max(np.abs(actual - expected))),
                           'top1_matches': bool(actual.argmax(1)[0] == expected.argmax(1)[0])})
    result = {**checkpoint_provenance(checkpoint, path), 'onnx_sha256': sha256(exported),
              'opset': 17, 'input_shape': [1, 1, 28, 28], 'rtol': 1e-3, 'atol': 1e-4,
              'checks': checks, 'environment': environment(), 'onnx': onnx.__version__,
              'onnxruntime': ort.__version__, 'providers': session.get_providers()}
    write_json(output / 'validation.json', result)
    print(f'PASS: ONNX structure and all {len(checks)} numerical comparisons. Saved {exported}')


if __name__ == '__main__':
    run_cli(main)
