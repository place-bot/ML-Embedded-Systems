"""Download pinned official detector weights and verify the bundled teaching files."""
import urllib.request
from common import ROOT, WEIGHTS_SHA256, sha256
import json


def main():
    target = ROOT/'data/yolov2.weights'
    target.parent.mkdir(exist_ok=True)
    if not target.exists() or sha256(target) != WEIGHTS_SHA256:
        temporary = target.with_suffix('.download')
        print('Downloading YOLOv2 weights (204 MB)...',flush=True)
        request = urllib.request.Request(
            'https://data.pjreddie.com/files/yolov2.weights',
            headers={'User-Agent': 'CSE60685-Lab4/1.0'})
        with urllib.request.urlopen(request,timeout=60) as response, temporary.open('wb') as f:
            while True:
                chunk = response.read(1024*1024)
                if not chunk: break
                f.write(chunk)
        if sha256(temporary) != WEIGHTS_SHA256:
            raise ValueError('Weight download checksum differs. Run this command again.')
        temporary.replace(target)
    manifest = json.loads((ROOT/'assets/inria/manifest.json').read_text())
    for row in manifest['files']:
        if sha256(ROOT/row['path']) != row['sha256']:
            raise ValueError('Course data changed: '+row['path'])
    print('Weights and teaching images verified.')


if __name__ == '__main__': main()
