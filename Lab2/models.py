"""Exercise: implement forward and supply two architecture configurations."""
import torch
from torch import nn

# Keep baseline unchanged. Add dictionaries with the same four keys for B and C.
MODEL_CONFIGS = {
    'baseline': {'c1': 6, 'c2': 16, 'h1': 120, 'h2': 84},
    'conv_small': {'c1': 3, 'c2': 8, 'h1': 120, 'h2': 84},
    'fc_small': {'c1': 6, 'c2': 16, 'h1': 60, 'h2': 42},
}


def validate_config(config, name=None):
    if not isinstance(config, dict) or set(config) != {'c1', 'c2', 'h1', 'h2'}:
        raise ValueError('An architecture needs exactly c1, c2, h1 and h2.')
    if any(type(value) is not int or value < 1 for value in config.values()):
        raise ValueError('Architecture widths must be positive integers.')
    if any(value > 1024 for value in config.values()):
        raise ValueError('Widths above 1024 are outside this small-network lab.')
    if name == 'baseline' and config != {'c1': 6, 'c2': 16, 'h1': 120, 'h2': 84}:
        raise ValueError('Keep the baseline architecture at 6/16 and 120/84.')
    if name == 'conv_small' and config != {'c1': 3, 'c2': 8, 'h1': 120, 'h2': 84}:
        raise ValueError('Conv-small must use channels 3/8 and hidden widths 120/84.')
    if name == 'fc_small' and (config['c1'] != 6 or config['c2'] != 16 or config['h1'] > 60 or config['h2'] > 42):
        raise ValueError('FC-small requires channels 6/16, h1 in 1..60 and h2 in 1..42.')
    return dict(config)


def get_config(name):
    if name not in MODEL_CONFIGS:
        raise ValueError('Model must be baseline, conv_small or fc_small.')
    if MODEL_CONFIGS[name] is None:
        raise NotImplementedError(f'Fill MODEL_CONFIGS[{name!r}] in models.py first.')
    return validate_config(MODEL_CONFIGS[name], name)


class LeNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = validate_config(config)
        c1, c2, h1, h2 = (self.config[key] for key in ('c1', 'c2', 'h1', 'h2'))
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, 5), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(c2 * 4 * 4, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(), nn.Linear(h2, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)
