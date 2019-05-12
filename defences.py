import math
import torch
from torch.optim import SGD
from tqdm import trange, tqdm
from torch.utils.data import TensorDataset, DataLoader

from utils import to, random_latents, setup_run
from modules import DCGenerator, ResNetGenerator, MLPAutoEncoder, CNNClassifier, MLPClassifier


class Defence:
    def __init__(self, is_gan=False):
        self.is_gan = is_gan

    def defence(self, model, attacked_data_loader):
        model = to(model).eval()
        defended_correct = 0
        total = 0
        base_correct = 0
        adl = tqdm(attacked_data_loader, total=5) if self.is_gan else attacked_data_loader
        for idx, (x, y) in enumerate(adl):
            with torch.no_grad():
                pred = model(x)
                pred = pred.argmax(dim=1)
                base_correct += pred.eq(y).sum().item()
            x, y = to(self._defence(x)), to(y)
            with torch.no_grad():
                pred = model(x)
                pred = pred.argmax(dim=1)
                defended_correct += pred.eq(y).sum().item()
                total += pred.size(0)
            if self.is_gan:
                adl.set_description('Def: {} / Base: {}'.format(defended_correct / total, base_correct / total))
                if idx >= 4:
                    break
        return defended_correct / total, base_correct / total

    def _defence(self, x):
        raise NotImplementedError()


class SequentialDefence(Defence):
    def __init__(self, *args: Defence):
        super().__init__()
        self.models = list(args)

    def _defence(self, x):
        o = x
        for model in self.models:
            o = model._defence(o)
        return o


class AutoEncoderDefence(Defence):
    def __init__(self):
        super().__init__()
        model = MLPAutoEncoder()
        model.load_state_dict(torch.load('./trained_models/mnist_ae_mlp.pt', map_location='cpu'))
        self.model = to(model).eval()

    def _defence(self, x):
        with torch.no_grad():
            return self.model((to(x) + 1) / 2) * 2 - 1


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


class GanDefence(Defence):
    def __init__(self, path, config: GeneratorConfig):
        super().__init__(is_gan=True)
        self.config = config
        common_nn_args = dict(rgb_channels=1, dim=config.model_dim, num_classes=10 if config.cond else -1)
        generator = DCGenerator if config.dcgan else ResNetGenerator
        generator = generator(**common_nn_args, z_dim=config.z_dim, apply_sn=False)
        generator.load_state_dict(torch.load(path, map_location='cpu'))
        generator = generator.eval()
        self.generator = to(generator)

    def _defence(self, x):
        conf = self.config
        batch_size = x.size(0)
        multiplier = conf.recon_restarts * (10 if conf.cond else 1)
        z = to(random_latents(batch_size * multiplier, conf.z_dim, conf.z_distribution))
        y = to(torch.arange(10).repeat_interleave(batch_size * conf.recon_restarts, dim=0))
        z = torch.nn.Parameter(z, requires_grad=True)
        optim = SGD([z], conf.recon_step_size)
        x = to(x.repeat_interleave(multiplier, dim=0))
        for _ in trange(conf.recon_iters):
            fake = self.generator(z, y)
            loss = ((x - fake) ** 2).mean(dim=[1, 2, 3])
            optim.zero_grad()
            loss.mean().backward()
            optim.step()
        fake.detach_()
        return torch.stack([fake[i * multiplier + loss[i * multiplier:(i + 1) * multiplier].argmin().item()]
                            for i in range(batch_size)], dim=0)


class FastGanDefence(Defence):
    def __init__(self, path, config: GeneratorConfig):
        super().__init__(is_gan=False)  # it's this fast!
        config.recon_restarts = max(32, config.recon_restarts)
        self.config = config
        common_nn_args = dict(rgb_channels=1, dim=config.model_dim, num_classes=10 if config.cond else -1)
        generator = DCGenerator if config.dcgan else ResNetGenerator
        generator = generator(**common_nn_args, z_dim=config.z_dim, apply_sn=False)
        generator.load_state_dict(torch.load(path, map_location='cpu'))
        generator = generator.eval()
        self.generator = to(generator)

    def _defence(self, x):
        conf = self.config
        batch_size = x.size(0)
        multiplier = conf.recon_restarts * (10 if conf.cond else 1)
        with torch.no_grad():
            z = to(random_latents(batch_size * multiplier, conf.z_dim, conf.z_distribution))
            y = to(torch.arange(10).repeat_interleave(batch_size * conf.recon_restarts, dim=0))
            x = to(x.repeat_interleave(multiplier, dim=0))
            fake = self.generator(z, y)
            loss = ((x - fake) ** 2).mean(dim=[1, 2, 3])
            return torch.stack([fake[i * multiplier + loss[i * multiplier:(i + 1) * multiplier].argmin().item()]
                                for i in range(batch_size)], dim=0)


class BinarizeDefence(Defence):
    def _defence(self, x):
        return (x.sign() + 0.01).sign().float().to(x)  # we don't want the 0 of sign()


class NoDefence(Defence):
    def _defence(self, x):
        return x


# borrowed from https://github.com/BorealisAI/advertorch/blob/master/advertorch/defenses/smoothing.py
class GaussianKernelDefence(Defence):
    def __init__(self, sigma=2, kernel_size=5):
        super().__init__()
        vecx = torch.arange(kernel_size).float()
        vecy = torch.arange(kernel_size).float()
        gridxy = self._meshgrid(vecx, vecy)
        mean = (kernel_size - 1) / 2.
        var = sigma ** 2
        gaussian_kernel = (1. / (2. * math.pi * var) * torch.exp(-(gridxy - mean).pow(2).sum(dim=0) / (2 * var)))
        gaussian_kernel /= torch.sum(gaussian_kernel)
        gaussian_kernel = gaussian_kernel.repeat(1, 1, 1, 1)
        kernel = gaussian_kernel
        channels = kernel.shape[0]
        kernel_size = kernel.shape[-1]
        filter_ = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                  groups=channels, padding=kernel_size // 2, bias=False)
        filter_.weight.data = kernel
        filter_.weight.requires_grad = False
        self.filter = to(filter_)

    def _defence(self, x):
        with torch.no_grad():
            return self.filter(x)

    @staticmethod
    def _meshgrid(vecx, vecy):
        gridx = vecx.repeat(len(vecy), 1)
        gridy = vecy.repeat(len(vecx), 1).t()
        return torch.stack([gridx, gridy])


def get_classifier(cnn, adv):
    classifier_path = './trained_models/mnist_{}{}.pt'.format('cnn' if cnn else 'mlp', '_adv' if adv else '')
    classifier = CNNClassifier if cnn else MLPClassifier
    classifier = classifier()
    classifier.load_state_dict(torch.load(classifier_path, map_location='cpu'))
    return to(classifier).eval()


def get_attacked_data_loader(cnn, attack_id: int, batch_size):  # TODO make attack_id an Enum(7 modes)
    data_path = './saved_attacks/{}_{}.pth'.format('cnn' if cnn else 'mlp',
                                                   ['default', 'fgsm_0.15', 'fgsm_0.3', 'rfgsm', 'cw2', 'bb_cnn',
                                                    'bb_mlp'][attack_id])
    data = torch.load(data_path)
    tensor_dataset = TensorDataset(to(data['x']), to(data['y']))
    return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def get_defence(defence_id: int):  # TODO make defence_id an Enum(7 modes)
    if defence_id == 0:
        return NoDefence()
    if defence_id == 1:
        return BinarizeDefence()
    if defence_id == 2:
        return GaussianKernelDefence()
    if defence_id == 3:
        return AutoEncoderDefence()
    if defence_id == 4:
        return SequentialDefence(AutoEncoderDefence(), BinarizeDefence())
    if defence_id == 5:
        return SequentialDefence(BinarizeDefence(), AutoEncoderDefence())
    default_gan_config = GeneratorConfig(model_dim=64, cond=False, dcgan=False, z_dim=100, recon_restarts=4,
                                         recon_iters=100, recon_step_size=0.5, z_distribution='normal')
    if defence_id == 6:
        return FastGanDefence('./trained_models/gan/4000.pth', default_gan_config)
    else:
        return GanDefence('./trained_models/gan/4000.pth', default_gan_config)
    # if defence_id <= 7:
    #     default_gan_config.recon_restarts = [4, 8][7-defence_id]
    #     return GanDefence('./trained_models/gan/4000.pth', default_gan_config)
    # if defence_id <= 10:
    #     default_gan_config.recon_iters = [100, 200, 400][10 - defence_id]
    #     return GanDefence('./trained_models/gan/4000.pth', default_gan_config)
    # if defence_id == 11:
    #     return GanDefence('./trained_models/gan/500.pth', default_gan_config)
    # if defence_id == 12:
    #     return SequentialDefence(BinarizeDefence(), GanDefence('./trained_models/gan/4000.pth', default_gan_config))
    # if defence_id == 13:
    #     return SequentialDefence(GanDefence('./trained_models/gan/4000.pth', default_gan_config), BinarizeDefence())


def main():
    # classifiers are in './trained_models/mnist_{}{}.pt'.format('mlp' or 'cnn', '_adv' or '')
    # attacks are in './saved_attacks/{}_{}.pth'.format('mlp' or 'cnn',
    #                           'default' or 'fgsm_0.15' or 'fgsm_0.3' or 'rfgsm' or 'cw2' or 'bb_cnn' or 'bb_mlp')
    # defences are Gan(cond or not, path=last or middle, resets=1 or 4 or 8, steps=100 or 200 or 400),
    #              AE(mlp or cnn), Bin, No, Gaussian, Seq(Bin, BestAE) + reverse, Seq(Bin, BestGan) + reverse
    # this would be 4 * 7(we use the same attack as classifier) * 45 = 1260 modes!

    # this is a short version (7 * 2 * 2 * 7)
    setup_run()
    idx = 0
    for defence_id in range(7):
        defence_mechanism = get_defence(defence_id)
        for classifier_arch in ('cnn', 'mlp'):
            for classifier_adv in (True, False):
                classifier = get_classifier(classifier_arch == 'cnn', classifier_adv)
                for attack_id in range(7):
                    attack_dl = get_attacked_data_loader(classifier_arch == 'cnn', attack_id, 128)
                    print(idx, '{}_{}_{}_{}:'.format(defence_id, classifier_arch, classifier_adv, attack_id),
                          defence_mechanism.defence(classifier, attack_dl))
                    idx += 1


if __name__ == '__main__':
    main()
