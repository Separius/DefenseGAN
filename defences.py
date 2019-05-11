import math
import torch
from torch.optim import SGD

from utils import to, random_latents
from modules import DCGenerator, ResNetGenerator, CNNAutoEncoder, MLPAutoEncoder


class Defence:
    def defence(self, model, attacked_data_loader):
        model = to(model).eval()
        correct = 0
        total = 0
        for x, y in attacked_data_loader:
            x, y = to(self._defence(x)), to(y)
            with torch.no_grad():
                pred = model(x)
                pred = pred.argmax(dim=1)
                correct += pred.eq(y).sum().item()
                total += pred.size(0)
        return correct / total

    def _defence(self, x):
        raise NotImplementedError()


class DefAE(Defence):
    def __init__(self, cnn):
        model = CNNAutoEncoder if cnn else MLPAutoEncoder
        model = model()
        model.load_state_dict(
            torch.load('./trained_models/mnist_ae_{}.pt'.format('cnn' if cnn else 'mlp'), map_location='cpu'))
        self.model = to(model).eval()

    def _defence(self, x):
        return self.model(to(x))


class GeneratorConfig:
    def __init__(self, model_dim, cond, dcgan, z_dim, recon_restarts, recon_iters, recon_step_size, z_distribution):
        self.model_dim = model_dim
        self.cond = cond
        self.dcgan = dcgan
        self.z_dim = z_dim
        self.recon_restarts = recon_restarts
        self.recon_iters = recon_iters
        self.recon_step_size = recon_step_size
        self.z_distribution = z_distribution


class DefGan(Defence):
    def __init__(self, path, config: GeneratorConfig):
        self.config = config
        common_nn_args = dict(rgb_channels=1, dim=config.model_dim, num_classes=10 if config.cond else -1)
        generator = DCGenerator if config.dcgan else ResNetGenerator
        generator = generator(**common_nn_args, z_dim=config.z_dim, apply_sn=False)
        generator.load_state_dict(torch.load(path, map_location='cpu')['g'])
        generator = generator.eval()
        self.generator = to(generator)

    def _defence(self, x):
        conf = self.config
        batch_size = x.size(0)
        z = to(random_latents(batch_size * conf.recon_restarts * (10 if conf.cond else 1), conf.z_dim,
                              conf.z_distribution))
        y = to(torch.arange(10).repeat_interleave(batch_size * conf.recon_restarts, dim=0))
        z = torch.nn.Parameter(z, requires_grad=True)
        optim = SGD([z], conf.recon_step_size)
        x = to(x.repeat_interleave(conf.recon_restarts, dim=0))
        for _ in range(conf.recon_iters):
            fake = self.generator(z, y)
            loss = ((x - fake) ** 2).mean(dim=[1, 2, 3])
            optim.zero_grad()
            loss.mean().backward()
            optim.step()
        fake.detach_()
        multiplier = conf.recon_restarts * (10 if conf.cond else 1)
        return torch.stack([fake[i * multiplier + loss[i * multiplier:(i + 1) * multiplier].argmin().item()]
                            for i in range(batch_size)], dim=0)


class Binarize(Defence):
    def _defence(self, x):
        return (x.sign() + 0.01).sign().float().to(x)  # we don't want the 0 of sign()


class NoDefence(Defence):
    def _defence(self, x):
        return x


# borrowed from https://github.com/BorealisAI/advertorch/blob/master/advertorch/defenses/smoothing.py
class GaussianKernel(Defence):
    def __init__(self, sigma=2, kernel_size=5):
        vecx = torch.arange(kernel_size).float()
        vecy = torch.arange(kernel_size).float()
        gridxy = self._meshgrid(vecx, vecy)
        mean = (kernel_size - 1) / 2.
        var = sigma ** 2
        gaussian_kernel = (1. / (2. * math.pi * var) * torch.exp(-(gridxy - mean).pow(2).sum(dim=0) / (2 * var)))
        gaussian_kernel /= torch.sum(gaussian_kernel)
        kernel = gaussian_kernel
        channels = kernel.shape[0]
        kernel_size = kernel.shape[-1]
        filter_ = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                  groups=channels, padding=kernel_size // 2, bias=False)
        filter_.weight.data = kernel
        filter_.weight.requires_grad = False
        self.filter = filter_

    def _defence(self, x):
        return self.filter(x)

    @staticmethod
    def _meshgrid(vecx, vecy):
        gridx = vecx.repeat(len(vecy), 1)
        gridy = vecy.repeat(len(vecx), 1).t()
        return torch.stack([gridx, gridy])
