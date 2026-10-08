"""Train only the patch pixels against the frozen YOLOv2 detector."""
import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from common import ROOT, CONFIG, WEIGHTS_SHA256, checked_weights, sha256, write_json
from yolo_model import YOLOv2
from patch_ops import apply_patch, patch_loss


def model_digest(model):
    import hashlib
    h = hashlib.sha256()
    for name,value in model.state_dict().items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_examples(split):
    manifest = json.loads((ROOT/'assets/inria/manifest.json').read_text())
    records = []
    for name in manifest[split]:
        image = np.array(Image.open(ROOT/'assets/inria/images'/name).convert('RGB'),dtype=np.float32)/255
        labels = np.loadtxt(ROOT/'assets/inria/labels'/(Path(name).stem+'.txt'),ndmin=2,dtype=np.float32)
        records.append((name,torch.from_numpy(image).permute(2,0,1),torch.from_numpy(labels)))
    return records


def save_png(patch, path):
    array = patch.detach().cpu().clamp(0,1).permute(1,2,0).numpy()
    Image.fromarray(np.round(array*255).astype(np.uint8)).save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--steps',type=int,default=CONFIG['steps'])
    parser.add_argument('--max-minutes',type=float,default=15)
    parser.add_argument('--output',type=Path,default=ROOT/'results/patch')
    parser.add_argument('--seed',type=int,default=CONFIG['seed'])
    args = parser.parse_args()
    if args.steps < 1 or args.max_minutes <= 0:
        parser.error('Use positive steps and minutes.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('No GPU is connected. Select a GPU in Colab, or use the reference path in the notebook.')
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(CONFIG['threads'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    model = YOLOv2(ROOT/'assets/cfg/yolov2.cfg').load_darknet_weights(checked_weights())
    model = model.to(args.device).eval().requires_grad_(False)
    before = model_digest(model)
    records = load_examples('train')
    data = torch.stack([r[1] for r in records]).to(args.device)
    count = max(len(r[2]) for r in records)
    labs = torch.ones(len(records),count,5,device=args.device)
    for i,record in enumerate(records): labs[i,:len(record[2])] = record[2].to(args.device)
    patch = torch.full((3,CONFIG['patch_pixels'],CONFIG['patch_pixels']),.5,device=args.device,requires_grad=True)
    colors = torch.tensor(np.loadtxt(ROOT/'assets/printable-colors.txt',delimiter=','),dtype=torch.float32,device=args.device)
    optimizer = torch.optim.Adam([patch],lr=CONFIG['learning_rate'],amsgrad=True)
    args.output.mkdir(parents=True,exist_ok=False)
    random_control = torch.rand(patch.shape,generator=torch.Generator().manual_seed(args.seed+1))
    save_png(random_control,args.output/'random.png')
    rows = []
    if args.device == 'cuda': torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in range(1,args.steps+1):
        indices = torch.randperm(len(records),device=args.device)[:CONFIG['batch_size']]
        image = apply_patch(data[indices],labs[indices],patch,augment=True)
        raw = model(image)
        loss,parts = patch_loss(raw,patch,colors)
        if not torch.isfinite(loss): raise RuntimeError('Training produced a non-finite loss.')
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        with torch.no_grad(): patch.clamp_(0,1)
        if args.device == 'cuda': torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        rows.append({'step':step,'seconds':elapsed,'loss':float(loss.detach()),
                     **{k:float(v.detach()) for k,v in parts.items()}})
        if step == 1 or step%100 == 0 or step == args.steps:
            save_png(patch,args.output/'learned.png')
            print(f'Step {step}/{args.steps}: objectness loss {rows[-1]["objectness"]:.3f}, {elapsed:.1f} s',flush=True)
        if elapsed >= args.max_minutes*60: break
    save_png(patch,args.output/'learned.png')
    after = model_digest(model)
    if before != after: raise RuntimeError('Detector weights changed.')
    with (args.output/'training.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    metadata = {'patch_source':'trained','model':CONFIG['model'],'weights_sha256':WEIGHTS_SHA256,
                'patch_sha256':sha256(args.output/'learned.png'),'random_sha256':sha256(args.output/'random.png'),
                'steps':step,'requested_steps':args.steps,'batch_size':CONFIG['batch_size'],'seed':args.seed,
                'elapsed_seconds':elapsed,'stopped_by_time_limit':step<args.steps,
                'model_unchanged':before==after,'model_state_sha256':after,
                'training_manifest_sha256':sha256(ROOT/'assets/inria/manifest.json'),
                'config':CONFIG,'device':torch.cuda.get_device_name() if args.device=='cuda' else platform.processor(),
                'peak_cuda_memory_mib':torch.cuda.max_memory_allocated()/1024**2 if args.device=='cuda' else None,
                'torch':torch.__version__,'python':platform.python_version()}
    write_json(args.output/'run.json',metadata)
    print(f'Saved {args.output}; detector unchanged: {before==after}',flush=True)


if __name__ == '__main__': main()
