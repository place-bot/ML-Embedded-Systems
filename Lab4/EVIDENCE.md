# Lab 4 evidence and reproduction

Training ran on a Colab Tesla T4. Camera inference ran on the supplied Raspberry Pi 4B Rev. 1.5 with an OV5647 camera, Raspberry Pi OS 64-bit / Debian 13.5, Python 3.13.5 and OpenCV 4.12.0. OMEN provided SSH, printing, backups, verification and report typesetting; no OMEN inference is reported.

## Retained results

| Condition | Person detected | Max person objectness | Inference (ms) | Updates/s | Original observation |
| --- | --- | ---: | ---: | ---: | --- |
| Clean | Yes | 0.857888699 | 1817.859 | 0.544975 | `002-clean` |
| Random | Yes | 0.614604771 | 1881.451 | 0.526783 | `003-random` |
| Learned | Yes | 0.861669183 | 1913.017 | 0.518137 | `006-learned` |
| Explore | Yes | 0.709441483 | 1996.866 | 0.496452 | `008-explore` |

The instructor program defines the score as maximum objectness among predictions whose top class is person, with threshold 0.40. It is not classification accuracy. All four saved observations still detect a person. Learned is almost unchanged from Clean; training by itself does not establish a physical suppression benefit.

Random was held higher and covered the face, while Learned was lower on the torso. Explore's recorded note is `turn slightly`; the image also contains motion blur and changed placement. The student chose to retain these observations. Their differences must not be interpreted as an isolated pattern or angle effect. The first `001-clean` capture contains a patch and is not the final Clean; all eight original observation JSON files remain retained to preserve the capture history. No observation was selected to fabricate an improvement.

## Evidence map

| Item | Location |
| --- | --- |
| Original instructor code, configuration, assets and licenses | This folder; original `README.md` and `NOTICE.md` unchanged |
| Source revision and per-file checksums | `provenance/experiment.json`, `provenance/source_sha256.json` |
| Actual trained and random patterns | `evidence/training/learned.png`, `random.png` |
| Complete 5,642-update training trace and frozen-detector record | `evidence/training/training.csv`, `training.json`, `run.json` |
| Four final camera rows and original observation metadata | `evidence/camera/comparison.csv`, `comparison.json`, `session.json`, numbered folders |
| Pi setup and trained-patch import checks | `provenance/pi_setup.log`, `pi_import_patch.log` |
| Report source and assistance disclosure | `report/Lab4_Report.tex`, `provenance/AI_ATTRIBUTION.md` |
| Public artifact byte-level manifest | `provenance/export_sha256.json` |
| Private/local artifact checksums | `provenance/retained_local_artifacts.json` |
| Read-only verification | `provenance/verify_evidence.py` |

The unmodified camera ZIP, all raw/annotated photographs, `evidence.jpg` and the illustrated report PDF are retained on OMEN. They contain personal photographs and are omitted from this public repository. The original patch ZIP and generated print PDFs are also retained locally; the public PNGs and supplied print script reproduce the paper pair. The repository's bundled instructor example/training images remain unchanged upstream assets with their notices.

The two-page `Lab4_Report.pdf` is the Canvas submission; source code and ZIPs are retained for inspection. The PDF includes the original four-panel evidence image, table, training/device details, discussion and concise AI attribution. It discloses the physical limitations. No Canvas submission or TA acceptance is claimed. After capture the student switched off power; an orderly OS shutdown was not verified for that session. The earlier setup session's successful shutdown is a separate event.

## Verification

Use standard Python; this performs no training or inference:

```bash
python Lab4/provenance/verify_evidence.py
```

It verifies published hashes, training records and camera metadata. Photos are omitted, so this public check cannot establish physical placement or resolve the disclosed experimental limitations. For a checkout that preserves original exported bytes, use `git -c core.autocrlf=false clone`.

## Repeat on the Raspberry Pi

Instructor source: https://github.com/guoyb17/CSE60685-FA26-Lab-4 at `5f27af02b15018e8ba98467f466023c014d8e5db`. All 192 instructor-tracked files are unchanged. The instructor notebook is https://colab.research.google.com/drive/1uUrYiNTPYdX4QeIqvOUeqetCpCwGP525 . The original training run used a 300 x 300 RGB patch, batch 4, learning rate 0.03, seed 42 and the default 6,000-update / 15-minute budget; it stopped at 900.030637021 seconds with 5,642 updates.

To use the retained trained pair without rerunning training, copy it to a fresh local run directory on the Pi:

```bash
cd Lab4
python3 -m venv env
source env/bin/activate
python -m pip install -r requirements.txt
python prepare_assets.py
python check_environment.py
rpicam-hello --list-cameras
mkdir -p results/patch-repeat
cp evidence/training/* results/patch-repeat/
python live_demo.py --patch results/patch-repeat
```

Forward the selected Pi's loopback port with `ssh -L 8080:127.0.0.1:8080 USER@PI_ADDRESS` from the control computer, then open `http://127.0.0.1:8080`. Use the actual camera, not `--image`. For a controlled repeat, keep pose/camera/lighting and patch location similar, wait for several detector updates after each change, and change one factor for Explore. Save/download all four conditions, retain the original records, stop the demo, back up results and shut down the OS before disconnecting power.

If the print pair needs regeneration, use the provided `make_prints.py --patch results/patch-repeat` in an environment with the instructor's ReportLab dependency. Each pattern is four Letter pages printed in color at actual size; measure the 20 cm tiles before assembling the nominal 40 cm square. This does not retrain or alter the patch PNGs.

## Rebuild the private report

The public LaTeX source is unchanged. Restore only the privately retained `evidence.jpg` to `Lab4/artifacts/camera-20261008-135913/evidence.jpg` (this directory is gitignored), then:

```bash
cd Lab4/report
mkdir -p ../output/pdf
xelatex -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=../output/pdf Lab4_Report.tex
```

Times New Roman and Consolas are needed. The output remains under the ignored `Lab4/output/` directory. The report's photo placement changes neither the image contents nor the recorded detection values.
