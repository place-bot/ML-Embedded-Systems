# CSE 60685 Fall 2026 Lab 2

Design and train small CNNs on Raspberry Pi.
Follow the accompanying Lab 2 Tutorial and Assignment.
Compare the three models using the supplied dataset split and training settings, with one CPU thread for inference.

## Installation on the Pi

After cloning this repo, continue in a Pi terminal:

```bash
cd ~/CSE60685-FA26-Lab-2
python3 -m venv env
source env/bin/activate
python -m pip install -r requirements.txt
python check_environment.py
python prepare_data.py
```

Use the Pi setup from Lab 1.
The code was tested on 64-bit Debian 13 (Trixie) with Python 3.13.5.
Run all required training and measurements on the Pi CPU.
Keep Lab 1's environment separate.
After reconnecting, return to this folder and run `source env/bin/activate` again.

The environment check uses its own tiny model, so it works before you complete the exercises.
`prepare_data.py` downloads the four official Fashion-MNIST files, verifies their checksums, and checks the published split.
Re-running it verifies existing files without replacing them.

## Complete the exercises

1. In `models.py`, complete `LeNet.forward`: pass the input through `self.features`, flatten from dimension 1, and pass the features through `self.classifier`. Return logits without softmax.
2. In `train_step.py`, replace the exception with `optimizer.zero_grad()`, then add `loss.backward()` and `optimizer.step()` after computing the loss.
3. Fill the two `None` entries in `MODEL_CONFIGS`. Keep A at channels 6/16 and hidden widths 120/84. B uses channels 3/8 and hidden widths 120/84. C keeps channels 6/16; choose positive integer hidden widths with h1 <= 60 and h2 <= 42.

The Tutorial provides the completed baseline forward and training step.

```bash
python check_model.py --model baseline
python train.py --model baseline --output results/baseline
python evaluate.py --checkpoint results/baseline/checkpoint.pt --verify-only --output results/baseline/reload_check
```

The reload check evaluates the validation split and does not evaluate the test split.
A checkpoint stores its own architecture configuration;
changing `MODEL_CONFIGS` later does not reinterpret older weights.

## Finish the three-model comparison

Inspect B and C with `check_model.py`, then train each from scratch:

```bash
python check_model.py --model conv_small
python check_model.py --model fc_small
python train.py --model conv_small --output results/conv_small
python train.py --model fc_small --output results/fc_small
```

All three use the fixed 12,000-image training subset, a disjoint 2,000-image validation subset, 5 epochs, batch size 64, Adam at learning rate 0.001 and seed 42.
CPU intra-op/inter-op thread counts are both one and DataLoader workers are zero.
No image augmentation or additional normalization is applied.
Each model starts from random initialization and uses the same shuffled training index order.

Choose all three configurations before running the final test evaluation:

```bash
for model in baseline conv_small fc_small; do
    python evaluate.py --checkpoint results/$model/checkpoint.pt --output results/$model/evaluation
done
```

Let the Pi cool to a comparable starting temperature before each benchmark.
Keep power, cooling and other activity unchanged, and run only one benchmark at a time:

```bash
for model in baseline conv_small fc_small; do
    python benchmark.py --checkpoint results/$model/checkpoint.pt --threads 1 --warmup 10 --runs 100 --output results/$model/benchmark
done
python summarize.py --runs results/baseline results/conv_small results/fc_small --output results/comparison
```

Pause between individual benchmark commands if needed to restore a comparable temperature;
the loop itself does not wait for cooling.
Device temperature and available power/throttling flags are sampled outside the timed calls.

## Read the results

| Metric | Definition |
| --- | --- |
| Parameters | All trainable weights and biases |
| Test accuracy (%) | Correct predictions divided by 10,000, multiplied by 100 |
| Mean training time (s/epoch) | Arithmetic mean over all five training loops, including data iteration and fixed batch statistics |
| Inference median (ms) | Median of 100 synchronous CPU model calls, batch size one |

Training times exclude validation, test evaluation, printing and saving.
Inference timing uses the same preloaded float32 validation tensor of shape `[1,1,28,28]` for every model, after `eval()` and inside `inference_mode()`.
It includes Python call overhead, but excludes loading, preprocessing, argmax/softmax, printing and file output.
There are 10 untimed warm-ups followed by exactly 100 CSV rows plus a header.
The input index and checksum appear in the benchmark JSON.

Each run contains `config.json`, `checkpoint.pt`, `training.csv`, `training_summary.json`, `evaluation/evaluation.json`, and `benchmark/summary.json` plus `benchmark/raw_timings.csv`.
`summarize.py` checks checkpoint hashes, settings, test counts and raw timing statistics before producing `comparison.csv` and `comparison.json`.

Commands with `--output` require a new destination.
If you repeat a run, use a fresh name and supply its matching paths to later commands.

## Optional exploration

Try four inference threads in a separate benchmark output directory, repeat training with more seeds in a separate exploratory copy, or inspect p95.
Keep the required settings for the main comparison.

ONNX export has separate dependencies:

```bash
python -m pip install -r requirements-optional.txt
python export_onnx.py --checkpoint results/baseline/checkpoint.pt --output results/baseline/onnx
```

This produces an opset-17 model with fixed batch size one and verifies zero, seeded random and three real validation inputs against PyTorch at rtol=1e-3 and atol=1e-4.
The pinned PyTorch exporter uses `dynamo=False`;
a legacy-export deprecation warning is expected.

## Report

Submit the PDF report described in the Assignment. Keep your code, checkpoints and logs for your records.

## Finish safely

Save and back up your work.
Run `sudo poweroff` on the Pi and wait for shutdown and SD-card activity to stop before switching off power, as in Lab 1.
