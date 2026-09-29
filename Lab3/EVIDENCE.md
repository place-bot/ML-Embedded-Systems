# Lab 3: structured pruning and fine-tuning

All model execution took place on the supplied Raspberry Pi 4 Model B Rev. 1.5 (aarch64), with Raspberry Pi OS 64-bit / Debian 13.5, Python 3.13.5 and PyTorch 2.8.0+cpu. The OMEN was used for SSH, file backup, read-only evidence checks and LaTeX typesetting. No OMEN model timings are reported.

## Submission

Canvas requires one report PDF. The final `results/Lab3_Report.pdf` contains the analysis and concise AI Attribution Appendix. The LaTeX source is retained in `report/Lab3_Report.tex`. Code, checkpoints, JSON/CSV and logs are retained here for inspection; they are not additional Canvas uploads unless the TA requests them.

The Canvas assignment page displayed October 4, 2026 at 11:59 p.m., whereas the Tutorial and Assignment PDFs said October 1 before class. The experiment was completed on September 29, before both dates.

## Actual results

| Model | Parameters | Validation before -> after fine-tuning (%) | Test (%) | Median inference (ms) |
| --- | ---: | ---: | ---: | ---: |
| Control | 44,426 | 79.65 -> 82.60 | 81.77 | 2.6906365 |
| S25 | 36,142 | 77.45 -> 82.35 | 81.70 | 2.6027935 |
| S50 | 27,858 | 64.05 -> 81.40 | 81.00 | 2.5075610 |

The source is the student's own last checkpoint from the required five-epoch Lab 2 A run, SHA-256 `47071f5d25f7aa5fef7414a187e6ea0ddf9ef12e01de022ff453a3d5ca9b870d`. It passed architecture, training-recipe, saved-logit and validation checks. No fallback checkpoint was used.

Control, S25 and S50 start independently from that source and each receive three additional epochs using the supplied recipe: Fashion-MNIST 12,000/2,000 train/validation split, batch 64, fresh Adam at 0.0001, seed 42, one intra-op/interop thread and zero loader workers. The last checkpoint is used. All training finished before complete 10,000-image test evaluations; all evaluations finished before timing.

S25/S50 retain 12/8 of 16 Conv2 output channels using L1 filter ranking and keep the associated blocks of 16 FC1 input columns. Conv1 and hidden widths 120/84 are unchanged. The compact models passed comparison against the masked-feature reference. Total parameter reductions are 18.65% and 37.29%, not 25% and 50%.

All benchmarks use the unchanged instructor program, the same preloaded float32 validation image of shape [1,1,28,28], one CPU thread, evaluation/inference mode, ten warmups and 100 recorded model calls. Runs were sequential; the supplied power adapter, closed case, running fan and ondemand governor were held unchanged. Each command was launched after cooling to at most 40 C; model/data loading raised the actual pre-warmup temperatures to 41.381-43.329 C. Sampled frequencies were 1.8 GHz and power/throttling flags were 0x0 at every benchmark boundary.

The source, training/evaluation results and timing results belong to one execution recorded in `results/setup/run_context.json`, completed at 2026-09-29 17:55:04 UTC. SSH interruptions did not terminate the detached Pi process. The student's later power cycle happened after completion. Wi-Fi power saving was disabled only after measurement and verified backup, as a connectivity troubleshooting step; the cause of disconnections was not conclusively established.

## Requirement-to-evidence map

| Requirement | Evidence |
| --- | --- |
| Separate environment, pinned dependencies, CPU/data checks | `results/setup/driver.log`, `python_packages.txt` |
| Own Lab 2 A imported and verified | `results/source/source.json`, `setup/import_baseline.log`, `setup/lab2_baseline_sha256.txt` |
| Read and test supplied channel selection/weight transfer | unchanged `pruning_ops.py`, `results/setup/check_pruning.log` |
| Three independent branches and immediate post-pruning validation | `results/pruned/{control,s25,s50}/pruning.json`, intermediate checkpoints |
| Three epochs per branch with fixed settings | each final model directory's `training.csv`, `config.json`, `training_summary.json`, `checkpoint.pt` |
| Reload verification and full test set | each `test/evaluation.json`, `results/setup/evaluate_*.log` |
| Same image, 1 thread, 10 warmups, 100 timings, sequential execution | each `timing/summary.json`, `timing/raw_timings.csv`, `results/setup/run_context.json` |
| Required summary table | `results/comparison/comparison.csv`, `.json`, `report_table.md` |
| Unchanged instructor scripts/assets and run provenance | `results/setup/upstream_commit.txt`, `reference_sha256.txt`, stage input/output hashes in `run_context.json` |
| Local verification without model execution | `provenance/evidence_audit.json`, `provenance/verify_evidence.py` (186 checks) |
| Report, personal decision and attribution | `results/Lab3_Report.pdf`, `report/Lab3_Report.tex`, `provenance/student-analysis.json`, `provenance/ai-assistance-log.md` |
| Verified experiment backup and operating-system shutdown | `provenance/05_collect_evidence.log`, `provenance/evidence_audit.json`, `provenance/shutdown.log` |

The original result files retain their exported bytes. The initial evidence archive SHA-256 is `dcc17b3091f35dbeb3181f9a7bd24fd1d7254ba94dee3d9da1d0435392336b0b`. The archive is backed up on OMEN. The published files have a separate manifest in `provenance/export_sha256.json`.

The operating-system poweroff command succeeded at 2026-09-29 18:09:22 UTC, followed by SSH disconnection. The student then confirmed that the green activity LED stopped. The report is typeset on OMEN after that experiment backup; its final document is retained on OMEN and in this repository, without restarting the Pi for document-only work.

## Source and reproduction

Instructor repository: <https://github.com/guoyb17/CSE60685-FA26-Lab-3>, commit `4c635083c39862381940db06d134bfe95ce92088`. All instructor Python files, configuration and assets are unchanged. Execution/cooling and evidence-audit helpers are separate under `provenance`; the repository's original README is retained.

The added `.gitattributes` files under `results`, `provenance` and `report` preserve exported bytes. The instructor's own root `.gitattributes` remains unchanged. For a new Windows checkout that also preserves all instructor metadata file bytes, clone with `git -c core.autocrlf=false clone https://github.com/place-bot/ML-Embedded-Systems.git`.

On the actual Pi, from the published `Lab3` folder, use a fresh results root so the retained run is never overwritten:

```bash
python3 -m venv env
source env/bin/activate
python -m pip install -r requirements.txt
python check_environment.py
python prepare_data.py
python import_baseline.py --checkpoint ../Lab2/results/baseline/checkpoint.pt --output results_repeat1/source
python check_pruning.py
python prune_models.py --baseline results_repeat1/source/checkpoint.pt --output results_repeat1/pruned
for model in control s25 s50; do
    python finetune.py --checkpoint results_repeat1/pruned/$model/checkpoint.pt --output results_repeat1/$model
done
for model in control s25 s50; do
    python evaluate.py --checkpoint results_repeat1/$model/checkpoint.pt --output results_repeat1/$model/test
done
# Cool to comparable starting temperatures before EACH separate benchmark.
python benchmark.py --checkpoint results_repeat1/control/checkpoint.pt --threads 1 --warmup 10 --runs 100 --output results_repeat1/control/timing
# Wait for the board to cool.
python benchmark.py --checkpoint results_repeat1/s25/checkpoint.pt --threads 1 --warmup 10 --runs 100 --output results_repeat1/s25/timing
# Wait for the board to cool.
python benchmark.py --checkpoint results_repeat1/s50/checkpoint.pt --threads 1 --warmup 10 --runs 100 --output results_repeat1/s50/timing
python summarize.py --results results_repeat1 --output results_repeat1/comparison
# Back up results before shutdown.
sudo poweroff
```

Wait for shutdown and SD-card activity to stop before disconnecting power. For the retained run's read-only evidence audit, run `python provenance/verify_evidence.py . --output /path/to/a/new/audit.json`; this audits the fixed recorded run, not a repeat.

To typeset the report with XeLaTeX, Times New Roman and Consolas installed:

```bash
cd report
xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=../results Lab3_Report.tex
```

Rebuilding changes document hashes; keep measurement files unchanged and update document entries in the manifest if publishing a rebuilt PDF.
