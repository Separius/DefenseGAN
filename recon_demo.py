import torch
import numpy as np
from tqdm import trange
from torch.optim import SGD
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torch.utils.data import DataLoader, TensorDataset

from defences import get_defence, get_classifier
from utils import to, random_latents, get_mnist_ds


def recon(gan_defence, x):  # NOTE this is a duplicate code
    conf = gan_defence.config
    conf.recon_restarts = 64
    conf.recon_iters = 1
    batch_size = x.size(0)
    z = to(random_latents(batch_size * conf.recon_restarts, conf.z_dim, conf.z_distribution))
    y = to(torch.arange(10).repeat_interleave(batch_size * conf.recon_restarts, dim=0))
    z = torch.nn.Parameter(z, requires_grad=True)
    optim = SGD([z], conf.recon_step_size)
    x_l = to(x.repeat_interleave(conf.recon_restarts, dim=0))
    for idx in trange(conf.recon_iters):
        fake = gan_defence.generator(z, y)
        loss = ((x_l - fake) ** 2).mean(dim=[1, 2, 3])
        optim.zero_grad()
        loss.mean().backward()
        optim.step()
        if idx % 10 == 0 or idx == (conf.recon_iters - 1):
            fake.detach_()
            x_r = torch.stack(
                [fake[i * conf.recon_restarts + loss[
                                                i * conf.recon_restarts:(i + 1) * conf.recon_restarts].argmin().item()]
                 for i in range(batch_size)], dim=0).cpu()
            plt.gcf()
            plt.subplot(1, 2, 1)
            plt.axis("off")
            plt.title("Real Images")
            plt.imshow(np.transpose(vutils.make_grid(x.cpu(), range=(0, 1.0), padding=5), (1, 2, 0)))
            plt.subplot(1, 2, 2)
            plt.axis("off")
            plt.title("Recon Images")
            plt.imshow(np.transpose(vutils.make_grid(x_r.cpu(), range=(0, 1.0), padding=5), (1, 2, 0)))
            plt.savefig('{}.png'.format(idx // 10))
    return x_r


def main():
    gan_defence = get_defence(float('+inf'))
    for x, y in DataLoader(get_mnist_ds(32, True), shuffle=True, batch_size=2):
        break
    tl = torch.load('./saved_attacks/cnn_cw2.pth')
    # tl = torch.load('./saved_attacks/cnn_fgsm_0.3.pth')
    for x, y in DataLoader(TensorDataset(tl['x'], tl['y']), shuffle=True, batch_size=4):
        break
    x_r = recon(gan_defence, x)
    classifier = get_classifier(cnn=True, adv=False)
    print('base', classifier(x).argmax(dim=1))
    print('defended', classifier(x_r).argmax(dim=1))
    print('truth', y)


if __name__ == '__main__':
    main()
