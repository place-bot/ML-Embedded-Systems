"""Compare fixed INRIA preview images with Clean, Random and Learned conditions."""
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from common import ROOT, configure, load_model, detect, annotate, write_json
from train_patch import load_examples
from patch_ops import apply_patch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--patch',type=Path,default=ROOT/'results/patch')
    args=p.parse_args()
    configure();torch.set_num_threads(2)
    net=load_model()
    patches={k:torch.from_numpy(np.array(Image.open(args.patch/(k.lower()+'.png')).convert('RGB'),dtype=np.float32)/255).permute(2,0,1) for k in ['Random','Learned']}
    rows=[];pages=[]
    for i,(name,image,labels) in enumerate(load_examples('preview')):
        if i%4==0:
            pages.append(Image.new('RGB',(1248,4*456),'white'))
        for j,condition in enumerate(['Clean','Random','Learned']):
            with torch.no_grad():
                tensor=image if condition=='Clean' else apply_patch(image[None],labels[None],patches[condition],False)[0]
            array=np.round(tensor.clamp(0,1).permute(1,2,0).numpy()*255).astype(np.uint8)
            example=Image.fromarray(array)
            result=detect(net,example)
            rows.append({'image':name,'condition':condition,**result})
            pages[-1].paste(annotate(example,result,'DIGITAL PREVIEW'),(j*416,(i%4)*456+40))
            ImageDraw.Draw(pages[-1]).text((j*416+8,(i%4)*456+10),f'{condition} | {name}',fill='black')
    write_json(args.patch/'preview.json',rows)
    for i,page in enumerate(pages):page.save(args.patch/('preview.jpg' if i==0 else f'preview-{i+1}.jpg'),quality=90)
    summary={condition:sum(r['person_detected'] for r in rows if r['condition']==condition) for condition in ['Clean','Random','Learned']}
    print('Preview images with a person detection:',summary)


if __name__=='__main__':main()
