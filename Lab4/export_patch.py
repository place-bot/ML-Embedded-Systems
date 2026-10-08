"""Package the selected patch pair and its training record for the Pi."""
import argparse
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from common import ROOT, patch_source_name
from live_demo import patch_metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--patch',type=Path,default=ROOT/'results/patch')
    p.add_argument('--output',type=Path,default=ROOT/'lab4-patch.zip')
    p.add_argument('--training',type=Path)
    p.add_argument('--no-gpu',action='store_true')
    args=p.parse_args()
    patch_metadata(args.patch)
    required=['learned.png','random.png','run.json','learned.pdf','random.pdf']
    if not all((args.patch/n).is_file() for n in required):
        raise ValueError('Run make_prints.py first.')
    selected = json.loads((args.patch/'run.json').read_text())
    training_folder = args.training or (args.patch if patch_source_name(selected['patch_source']) == 'trained' else None)
    training = json.loads((training_folder/'run.json').read_text()) if training_folder else {'status':'not_run','reason':'No Colab GPU' if args.no_gpu else 'Training record not supplied'}
    with ZipFile(args.output,'w',ZIP_DEFLATED) as z:
        z.writestr('training.json',json.dumps(training,indent=2)+'\n')
        for item in sorted(args.patch.iterdir()):
            if item.is_file() and item.name != 'training.json' and item.suffix in ['.png','.pdf','.json','.csv','.jpg']:
                z.write(item,item.name)
        if args.training and args.training.resolve()!=args.patch.resolve() and (args.training/'training.csv').exists():
            z.write(args.training/'training.csv','training.csv')
    print(args.output)


if __name__=='__main__':main()
