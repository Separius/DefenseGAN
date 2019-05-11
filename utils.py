import os
import torch
import random
import numpy as np
from copy import deepcopy
from typing import TypeVar
from pickle import load, dump
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST

EPSILON = 1e-8
_half_tensor = None

T = TypeVar('T')


def to(thing: T, device=None) -> T:
    if thing is None:
        return None
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(thing, (list, tuple)):
        return [to(item, device) for item in thing]
    if isinstance(thing, dict):
        return {k: to(v, device) for k, v in thing.items()}
    return thing.to(device)


def get_half(num_latents, latent_size):
    global _half_tensor
    if _half_tensor is None or _half_tensor.size() != (num_latents, latent_size):
        _half_tensor = torch.ones(num_latents, latent_size) * 0.5
    return _half_tensor


def random_latents(batch_size, z_dim, z_distribution='normal'):
    if z_distribution == 'normal':
        return torch.randn(batch_size, z_dim)
    elif z_distribution == 'censored':
        return F.relu(torch.randn(batch_size, z_dim))
    elif z_distribution == 'bernoulli':
        return torch.bernoulli(get_half(batch_size, z_dim))
    elif z_distribution == 'uniform':
        return torch.rand(batch_size, z_dim)
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


def trainable_params(model):
    return filter(lambda p: p.requires_grad, model.parameters())


def num_params(net):
    model_parameters = trainable_params(net)
    return sum([np.prod(p.size()) for p in model_parameters])


def setup_run(deterministic=False, given_seed=None):
    manual_seed = random.randint(0, 1023) if given_seed is None else given_seed
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    np.random.seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(manual_seed)
        torch.backends.cudnn.benchmark = not deterministic
        torch.backends.cudnn.deterministic = deterministic
    return manual_seed


def load_model(model_path):
    return torch.load(model_path, map_location='cpu')


def get_mnist_ds(size=64, train=True):
    return MNIST('~/.torch/data/', train=train, download=True, transform=transforms.Compose(
        [transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]))


def infinite_sampler(loader):
    while True:
        for b in loader:
            yield b


def flatten_params(model):
    return deepcopy(list(p.data for p in model.parameters()))


def load_params(flattened, model):
    for p, avg_p in zip(model.parameters(), flattened):
        p.data.copy_(avg_p)


def update_flattened(model, flattened, old_weight=0.99):
    for p, avg_p in zip(model.parameters(), flattened):
        avg_p.mul_(old_weight).add_((1.0 - old_weight) * p.data)
