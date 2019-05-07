import os
import torch
import numpy as np
from typing import TypeVar
from pickle import load, dump
import torch.nn.functional as F

EPSILON = 1e-8
_half_tensor = None

T = TypeVar('T')


def to(thing: T, device=None) -> T:
    if thing is None:
        return None
    if device is None:
        device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')
    if isinstance(thing, (list, tuple)):
        return [item.to(device) for item in thing]
    if isinstance(thing, dict):
        return {k: v.to(device) for k, v in thing.items()}
    return thing.to(device)


def random_latents(num_latents, latent_size, z_distribution='normal'):
    if z_distribution == 'normal':
        return torch.randn(num_latents, latent_size)
    elif z_distribution == 'censored':
        return F.relu(torch.randn(num_latents, latent_size))
    elif z_distribution == 'bernoulli':
        return torch.bernoulli(get_half(num_latents, latent_size))
    else:
        raise ValueError()


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_pkl(file_name, obj):
    with open(file_name, 'wb') as f:
        dump(obj, f, protocol=4)


def load_pkl(file_name):
    if not os.path.exists(file_name):
        return None
    with open(file_name, 'rb') as f:
        return load(f)


def get_half(num_latents, latent_size):
    global _half_tensor
    if _half_tensor is None or _half_tensor.size() != (num_latents, latent_size):
        _half_tensor = torch.ones(num_latents, latent_size) * 0.5
    return _half_tensor


def trainable_params(model):
    return filter(lambda p: p.requires_grad, model.parameters())


def num_params(net):
    model_parameters = trainable_params(net)
    return sum([np.prod(p.size()) for p in model_parameters])


def enable_benchmark():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def load_model(model_path, return_all=False):
    state = torch.load(model_path, map_location='cpu')
    if not return_all:
        return state['model']
    return state['model'], state['optimizer']