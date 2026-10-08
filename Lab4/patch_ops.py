"""Printable patch transformations adapted from EAVISE adversarial-yolo (MIT)."""
import math
import torch
from torch.nn import functional as F


def median_filter(patch):
    padded = F.pad(patch[None], (3,3,3,3), mode='reflect')
    return padded.unfold(2,7,1).unfold(3,7,1).flatten(-2).median(-1).values[0]


def apply_patch(images, labels, patch, augment=True):
    """Place one patch per valid person; labels are normalized cx,cy,w,h."""
    b,people,_ = labels.shape
    size = images.shape[-1]
    pixels = patch.shape[-1]
    filtered = median_filter(patch)
    patterns = filtered[None,None].expand(b,people,-1,-1,-1)
    if augment:
        contrast = torch.empty(b,people,1,1,1,device=patch.device).uniform_(.8,1.2)
        brightness = torch.empty_like(contrast).uniform_(-.1,.1)
        noise = torch.empty_like(patterns).uniform_(-.1,.1)
        patterns = (patterns*contrast+brightness+noise).clamp(.000001,.999999)
    else:
        patterns = patterns.clamp(.000001,.999999)
    mask = (labels[:,:,0] == 0).to(patch.dtype)[:,:,None,None,None].expand_as(patterns)
    pad = (size-pixels)//2
    rest = size-pixels-pad
    patterns = F.pad(patterns.reshape(-1,3,pixels,pixels),(rest,pad,rest,pad))
    mask = F.pad(mask.reshape(-1,3,pixels,pixels),(rest,pad,rest,pad))
    flat = labels.reshape(-1,5)
    scale = .2*torch.sqrt(flat[:,3].square()+flat[:,4].square())*size/pixels
    scale = scale.clamp_min(.001)
    angle = torch.zeros(b*people,device=patch.device)
    if augment:
        angle.uniform_(-math.pi/9,math.pi/9)
    cx,cy = flat[:,1],flat[:,2]-.05
    tx,ty = (0.5-cx)*2,(0.5-cy)*2
    co,si = angle.cos()/scale,angle.sin()/scale
    theta = torch.stack((co,si,tx*co+ty*si,-si,co,-tx*si+ty*co),dim=1).reshape(-1,2,3)
    grid = F.affine_grid(theta,patterns.shape,align_corners=False)
    placed = F.grid_sample(patterns,grid,align_corners=False).clamp(.000001,.999999)
    placed_mask = F.grid_sample(mask,grid,align_corners=False)
    placed = (placed*placed_mask).reshape(b,people,3,size,size)
    for person in range(people):
        overlay = placed[:,person]
        images = torch.where(overlay == 0,images,overlay)
    return images


def patch_loss(raw, patch, printable):
    head = raw.reshape(raw.shape[0],5,85,-1)
    detection = head[:,:,4].sigmoid().flatten(1).max(1).values.mean()
    delta = patch[None]-printable[:,:,None,None]+1e-6
    nps = (delta.square().sum(1)+1e-6).sqrt().min(0).values.sum()/patch.numel()
    tv = ((patch[:,1:]-patch[:,:-1]+1e-6).abs().sum()+
          (patch[:,:,1:]-patch[:,:,:-1]+1e-6).abs().sum())/patch.numel()
    total = detection+.01*nps+torch.clamp(2.5*tv,min=.1)
    return total, {'objectness':detection,'nps':nps,'tv':tv}
