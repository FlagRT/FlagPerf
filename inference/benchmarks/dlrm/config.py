"""Architecture checks run on the launcher without importing execution packages."""
from pathlib import Path
from runtime.common import read_json

DEFAULTS = {'dense_dim': 13, 'field_dims': [1000] * 26, 'embed_dim': 16,
            'bottom_mlp_dims': [64, 16], 'top_mlp_dims': [64, 32], 'interaction': 'dot'}


def validate_architecture(value):
    if not isinstance(value, dict) or set(value) != set(DEFAULTS):
        raise ValueError('invalid DLRM architecture keys')
    if (any(type(value[k]) is not int or value[k] < 1 for k in ('dense_dim', 'embed_dim'))
            or any(not isinstance(value[k], list) or not value[k]
                   or any(type(n) is not int or n < 1 for n in value[k])
                   for k in ('field_dims', 'bottom_mlp_dims', 'top_mlp_dims'))
            or value['bottom_mlp_dims'][-1] != value['embed_dim']
            or value['interaction'] not in ('dot', 'cat')):
        raise ValueError('invalid DLRM architecture dimensions/interaction')
    return value


def architecture(path):
    return validate_architecture(read_json(Path(path) / 'config.json'))


def interaction_width(config):
    features, width = len(config['field_dims']) + 1, config['embed_dim']
    return width + features*(features-1)//2 if config['interaction'] == 'dot' else features*width


def validate_tp_architecture(config, degree=2):
    widths = [config['dense_dim'], *config['bottom_mlp_dims'][:-1],
              interaction_width(config), *config['top_mlp_dims']]
    if degree != 2 or config['embed_dim'] % degree or any(width < degree for width in widths):
        raise ValueError('DLRM TP=2 requires even embedding dimension and at least two input columns per Linear')
