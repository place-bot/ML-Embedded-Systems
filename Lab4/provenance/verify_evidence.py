"""Read-only public-artifact checks. Does not evaluate physical camera conditions."""
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
checks = 0


def require(condition, description):
    global checks
    if not condition:
        raise ValueError(description)
    checks += 1


def load(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    exported = load('provenance/export_sha256.json')
    for relative, expected in exported.items():
        path = ROOT / relative
        require(path.is_file() and digest(path) == expected, 'Export hash: ' + relative)
    source = load('provenance/source_sha256.json')
    require(len(source) == 192, 'All instructor-tracked files retained')
    for relative, expected in source.items():
        require(digest(ROOT / relative) == expected, 'Instructor file unchanged: ' + relative)

    training = load('evidence/training/training.json')
    require(training == load('evidence/training/run.json'), 'Training and run records match')
    require(training['patch_source'] == 'trained' and training['model_unchanged'], 'Trained patch; detector frozen')
    require(training['config'] == load('course_config.json'), 'Instructor settings unchanged')
    require(training['device'] == 'Tesla T4' and training['steps'] == 5642, 'Recorded training device and updates')
    require(training['requested_steps'] == 6000 and training['stopped_by_time_limit'], 'Default training budget')
    require(900 <= training['elapsed_seconds'] < 930, 'Default time limit reached')
    for name, key in [('learned', 'patch_sha256'), ('random', 'random_sha256')]:
        require(digest(ROOT / f'evidence/training/{name}.png') == training[key], name + ' image hash')
    require(digest(ROOT / 'assets/inria/manifest.json') == training['training_manifest_sha256'], 'Training dataset manifest')
    with (ROOT / 'evidence/training/training.csv').open(newline='') as stream:
        trace = list(csv.DictReader(stream))
    require([int(row['step']) for row in trace] == list(range(1, 5643)), 'Every training update retained')
    require(all(math.isfinite(float(value)) for row in trace for value in row.values()), 'Finite training trace')
    require(abs(float(trace[-1]['seconds']) - training['elapsed_seconds']) < 1e-6, 'Final training time matches')

    session = load('evidence/camera/session.json')
    require(session['source'] == 'camera' and session['patch_source'] == 'trained', 'Actual camera source')
    require(session['config'] == training['config'], 'Recorded evaluation settings')
    require(session['training'] == training, 'Camera uses this trained patch record')
    rows = load('evidence/camera/comparison.json')
    require([row['condition'] for row in rows] == ['Clean', 'Random', 'Learned', 'Explore'], 'Four conditions')
    require([row['folder'] for row in rows] == ['002-clean', '003-random', '006-learned', '008-explore'], 'Final selected original records')
    with (ROOT / 'evidence/camera/comparison.csv').open(newline='') as stream:
        csv_rows = list(csv.DictReader(stream))
    require(len(csv_rows) == 4, 'Four CSV rows')
    for row, csv_row in zip(rows, csv_rows):
        original = load('evidence/camera/' + row['folder'] + '/observation.json')
        require(row == original, row['condition'] + ' summary equals original')
        require(row['person_detected'] is True and 0.4 <= row['max_person_score'] <= 1, row['condition'] + ' detection/score')
        require(row['inference_ms'] > 0 and row['update_fps'] > 0 and row['saved_unix'] >= row['captured_unix'], row['condition'] + ' timing')
        require(csv_row['condition'] == row['condition'] and float(csv_row['max_person_score']) == row['max_person_score'], row['condition'] + ' CSV score')
    require(rows[-1]['note'] == 'turn slightly', 'Original Explore note retained')
    print(json.dumps({'checks_passed': checks, 'inference_performed': False,
                      'physical_conditions': 'Not established by metadata checks; see disclosed limitations in EVIDENCE.md.',
                      'personal_photos': 'Retained locally; not present in the public handoff.'}, indent=2))


if __name__ == '__main__':
    main()
