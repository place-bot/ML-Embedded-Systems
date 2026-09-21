# Lab 2: retained experiment evidence

All required training, evaluation and inference ran on the supplied Raspberry Pi 4 Model B Rev 1.5 on September 21, 2026. OMEN was used for SSH, file backups, source comparison and arithmetic verification only. There are no OMEN model timings in the results.

## Course source and exercise changes

The original instructor [repository](https://github.com/guoyb17/CSE60685-FA26-Lab-2) is retained at commit `ef9e3787250eb3dfef7a14a3e79821c55a302256`; its original `README.md` remains intact. Among the instructor files, only `models.py` and `train_step.py` were changed. The forward and training-step bodies were transcribed from Tutorial Steps 2 and 3. B uses the prescribed widths; the student selected C's hidden widths of 60 and 42.

- [Changes to the two exercise files](results/setup/exercise_changes.patch)
- [Designs fixed before training and test evaluation](results/setup/designs_fixed.json)
- [Environment check](results/setup/environment_check.txt), [A check](results/setup/check_baseline.txt), [B check](results/setup/check_conv_small.txt), [C check](results/setup/check_fc_small.txt)
- [Source hashes](results/setup/reference_sha256.txt), [package versions](results/setup/python_packages.txt)

## Measurement conditions

Raspberry Pi OS 64-bit, Debian 13 trixie; Python 3.13.5; PyTorch 2.8.0+cpu; NumPy 2.2.6. Supplied CanaKit power supply, closed case and running fan, unchanged during the experiment; `ondemand` CPU governor. Required scripts were unchanged. Each model was trained from scratch for five epochs with batch size 64, Adam learning rate 0.001, seed 42, one intra-op and one inter-op thread and zero data-loader workers.

The supplied Fashion-MNIST split provided 12,000 training and 2,000 validation images; every model was evaluated on all 10,000 official test images after all three designs were fixed. The last-epoch checkpoints were reloaded and verified; A additionally has a separate verify-only run before test evaluation.

All training completed before any required inference benchmark. Benchmarks ran sequentially on the same fixed validation tensor, float32 `[1,1,28,28]`, with one CPU thread, 10 untimed warmups and 100 timed calls. Each raw inference CSV contains exactly 100 observations plus its header. Model execution includes Python-call overhead; input loading/preprocessing, argmax/softmax and output are outside the timer. Training times are arithmetic means of all five timed training loops, excluding validation, test and file output.

Each benchmark process was launched after cooling to 39.92 C. The instructor's snapshots immediately before warmup were 44.303 C (A), 43.816 C (B) and 42.842 C (C); loading/importing occurs between launch and those snapshots. All recorded power/throttling flags were `0x0`. These are one standard run per model; the 100 calls are repeated inference timings, not 100 independently trained models.

## Original results

The instructor's `summarize.py` passed checkpoint, recipe, split, input and statistic consistency checks and created [comparison.csv](results/comparison/comparison.csv) and [comparison.json](results/comparison/comparison.json).

| Model | Checkpoint | Training history | Test evaluation | Inference summary | All inference samples |
| --- | --- | --- | --- | --- | --- |
| A | [PT](results/baseline/checkpoint.pt) | [CSV](results/baseline/training.csv) | [JSON](results/baseline/evaluation/evaluation.json) | [JSON](results/baseline/benchmark/summary.json) | [CSV](results/baseline/benchmark/raw_timings.csv) |
| B | [PT](results/conv_small/checkpoint.pt) | [CSV](results/conv_small/training.csv) | [JSON](results/conv_small/evaluation/evaluation.json) | [JSON](results/conv_small/benchmark/summary.json) | [CSV](results/conv_small/benchmark/raw_timings.csv) |
| C | [PT](results/fc_small/checkpoint.pt) | [CSV](results/fc_small/training.csv) | [JSON](results/fc_small/evaluation/evaluation.json) | [JSON](results/fc_small/benchmark/summary.json) | [CSV](results/fc_small/benchmark/raw_timings.csv) |

Retained provenance includes the [ordered stage record](results/setup/run_context.json), [console log](results/setup/experiment_driver.log), [original Pi file hashes](results/setup/evidence_sha256.json), and [144 passed export/source/measurement checks](provenance/evidence_audit.json). The audit compares source files to canonical upstream Git blobs, avoiding Windows CRLF checkout differences. All original measurement files preserve their exact exported bytes.

The [orchestration helper](provenance/run_required_experiments.py) was AI-written operational support running outside the instructor repository; it called unchanged instructor programs, recorded stages and paused for cooling. SSH was detached from the experiment so Wi-Fi disconnections did not terminate training. See the [AI assistance record](provenance/ai-assistance-log.md). The raw Fashion-MNIST download and virtual environment are excluded from Git; `prepare_data.py` downloads and verifies the official dataset again.

## Reproduce on the Pi

Follow the instructor's retained `README.md` to create the virtual environment, install `requirements.txt`, run `check_environment.py` and `prepare_data.py`, and run the three model checks. Use fresh output folders so these recorded results remain unchanged:

```bash
python train.py --model baseline --output results_repeat/baseline
python evaluate.py --checkpoint results_repeat/baseline/checkpoint.pt --verify-only --output results_repeat/baseline/reload_check
# Cool to a comparable starting temperature before each next training run.
python train.py --model conv_small --output results_repeat/conv_small
python train.py --model fc_small --output results_repeat/fc_small
```

After all training completes, run `evaluate.py` on each last-epoch checkpoint into its `evaluation` subfolder. Then cool and benchmark each checkpoint sequentially, using `--threads 1 --warmup 10 --runs 100` and its `benchmark` subfolder. Finally run `summarize.py --runs results_repeat/baseline results_repeat/conv_small results_repeat/fc_small --output results_repeat/comparison`.

Canvas submission is [Lab2_Report.pdf](results/Lab2_Report.pdf): one page of analysis plus the required AI Attribution Appendix, which is excluded from the analysis limit. The report follows the student's choice of C and interpretation of the architecture/runtime tradeoff. Code, checkpoints and raw results are retained for inspection. [LaTeX source](report/Lab2_Report.tex) and [document hashes](results/report_manifest.json) are retained separately from the original measurement manifest. Rebuild with XeLaTeX, Times New Roman and Consolas; recompilation may change the PDF hash but must not alter the measurement evidence.

## Backup and safe shutdown

The final PDF and LaTeX source were copied back to the Pi and verified alongside all original experiment files; see [final Pi verification](provenance/final_pi_verification.log). A complete archive (excluding the reinstallable environment and dataset) was downloaded to OMEN and its SHA-256 matched the Pi: `a08273ef52fe48f8a7370af412ed5e6f1107fdd174dac872268b755053508ac2`.

At September 21, 2026, 22:28:27 UTC, `poweroff` was invoked through `sudo`, returned successfully with exit code 0, and SSH subsequently disconnected. The student then confirmed that the green activity light had stopped flashing. The [shutdown log](provenance/shutdown.log) is retained. The report has been prepared for the student's Canvas upload; no Canvas submission was made by the assistant.
