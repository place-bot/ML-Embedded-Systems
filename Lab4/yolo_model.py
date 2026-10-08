"""YOLOv2 layers and Darknet weight reader used for patch gradients.

Adapted from EAVISE adversarial-yolo / marvis pytorch-yolo2 (MIT).
Only the layers in the supplied, unmodified COCO configuration are needed.
"""
from pathlib import Path
import numpy as np
import torch
from torch import nn


def read_cfg(path):
    blocks = []
    for line in Path(path).read_text().splitlines():
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        if line.startswith('['):
            blocks.append({'type': line[1:-1]})
        else:
            key, value = line.split('=', 1)
            blocks[-1][key.strip()] = value.strip()
    return blocks


class Reorg(nn.Module):
    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h//2, 2, w//2, 2).transpose(3, 4).contiguous()
        x = x.reshape(b, c, h*w//4, 4).transpose(2, 3).contiguous()
        x = x.reshape(b, c, 4, h//2, w//2).transpose(1, 2).contiguous()
        return x.reshape(b, 4*c, h//2, w//2)


class YOLOv2(nn.Module):
    def __init__(self, cfg_path):
        super().__init__()
        self.blocks = read_cfg(cfg_path)[1:]
        self.layers = nn.ModuleList()
        channels = []
        current = 3
        self.routes = {}
        for i, block in enumerate(self.blocks):
            kind = block['type']
            if kind == 'convolutional':
                count, size = int(block['filters']), int(block['size'])
                bn = block.get('batch_normalize', '0') == '1'
                layer = [nn.Conv2d(current, count, size, int(block['stride']),
                                   (size-1)//2 if int(block['pad']) else 0, bias=not bn)]
                if bn:
                    layer.append(nn.BatchNorm2d(count))
                if block['activation'] == 'leaky':
                    layer.append(nn.LeakyReLU(0.1, inplace=True))
                elif block['activation'] != 'linear':
                    raise ValueError('Unsupported activation')
                current = count
                self.layers.append(nn.Sequential(*layer))
            elif kind == 'maxpool':
                self.layers.append(nn.MaxPool2d(int(block['size']), int(block['stride'])))
            elif kind == 'reorg':
                assert int(block['stride']) == 2
                self.layers.append(Reorg())
                current *= 4
            elif kind == 'route':
                indices = [int(v) for v in block['layers'].split(',')]
                indices = [v if v >= 0 else i+v for v in indices]
                self.routes[i] = indices
                current = sum(channels[v] for v in indices)
                self.layers.append(nn.Identity())
            elif kind == 'region':
                self.layers.append(nn.Identity())
            else:
                raise ValueError('Unsupported layer: ' + kind)
            channels.append(current)

    def forward(self, x):
        outputs = []
        for i, layer in enumerate(self.layers):
            if i in self.routes:
                parts = [outputs[j] for j in self.routes[i]]
                x = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
            else:
                x = layer(x)
            outputs.append(x)
        return x

    def load_darknet_weights(self, path):
        with open(path, 'rb') as handle:
            header = np.fromfile(handle, dtype=np.int32, count=4)
            if tuple(header[:3]) != (0, 1, 0):
                raise ValueError('Expected the pinned YOLOv2 weight format.')
            values = np.fromfile(handle, dtype=np.float32)
        offset = 0
        with torch.no_grad():
            for block, layer in zip(self.blocks, self.layers):
                if block['type'] != 'convolutional':
                    continue
                conv = layer[0]
                if block.get('batch_normalize', '0') == '1':
                    bn = layer[1]
                    tensors = [bn.bias, bn.weight, bn.running_mean, bn.running_var, conv.weight]
                else:
                    tensors = [conv.bias, conv.weight]
                for tensor in tensors:
                    count = tensor.numel()
                    tensor.copy_(torch.from_numpy(values[offset:offset+count].copy()).reshape_as(tensor))
                    offset += count
        if offset != len(values):
            raise ValueError('Weight count differs from the model configuration.')
        return self
