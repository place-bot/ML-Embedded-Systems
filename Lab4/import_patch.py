"""Validate a Colab patch ZIP and install it for the camera session."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from zipfile import ZipFile
from common import ROOT,WEIGHTS_SHA256,sha256,patch_source_name


def install(archive,output):
    import hashlib
    from PIL import Image
    import io
    with ZipFile(archive) as z:
        infos=z.infolist()
        if len(infos)>30 or sum(i.file_size for i in infos)>64*1024*1024:
            raise ValueError('Unexpected patch ZIP size.')
        names=[i.filename for i in infos]
        if len(set(names))!=len(names) or any(Path(n).name!=n or '/' in n or '\\' in n for n in names):
            raise ValueError('Use an unmodified Colab patch ZIP.')
        required=['learned.png','random.png','run.json']
        if not set(required)<=set(names):raise ValueError('Patch ZIP is missing required files.')
        content={n:z.read(n) for n in names}
    run=json.loads(content['run.json'])
    if run['weights_sha256']!=WEIGHTS_SHA256:raise ValueError('This patch uses a different detector.')
    for name,key in [('learned.png','patch_sha256'),('random.png','random_sha256')]:
        if hashlib.sha256(content[name]).hexdigest()!=run[key]:raise ValueError('Patch checksum differs.')
        with Image.open(io.BytesIO(content[name])) as im:
            if im.size!=(300,300) or im.mode!='RGB':raise ValueError('Expected a 300 x 300 RGB patch.')
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        backup=output.with_name(output.name+'-backup-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'))
        output.rename(backup)
    output.mkdir()
    for name,data in content.items():(output/name).write_bytes(data)
    return run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('archive',type=Path)
    p.add_argument('--output',type=Path,default=ROOT/'results/patch')
    args=p.parse_args()
    run=install(args.archive,args.output)
    print(f'Installed {patch_source_name(run["patch_source"])} patch in {args.output}')


if __name__=='__main__':main()
