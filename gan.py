import torch
import numpy as np
from torch.optim import Adam
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from fid import prepare_inception_metrics
from losses import discriminator_loss, generator_loss
from modules import DCGenerator, DCDiscriminator, ResNetGenerator, ResNetDiscriminator
from utils import (infinite_sampler, random_latents, flatten_params, update_flattened,
                   to, num_params, trainable_params, get_mnist_ds, setup_run, load_params)


def train_gan(args):
    args['seed'] = setup_run(args['deterministic'])
    if args['cond']:
        assert args['grad_lambda'] == 0
    if args['verbose']:
        print(args)
    if args['tensorboard']:
        writer = SummaryWriter(args['experiment_name'] if args['experiment_name'] != '' else None)
    common_nn_args = dict(rgb_channels=1, dim=args['model_dim'], num_classes=10 if args['cond'] else -1)
    if args['dcgan']:
        generator, discriminator = DCGenerator, DCDiscriminator
    else:
        generator, discriminator = ResNetGenerator, ResNetDiscriminator
    generator = to(generator(**common_nn_args, z_dim=args['z_dim'], apply_sn=False))
    discriminator = to(discriminator(**common_nn_args, apply_bn=args['bn_in_d'], apply_sn=not args['no_spectral_norm']))
    if args['verbose']:
        print('generator\'s num params:', num_params(generator))
        print('discriminator\'s num params:', num_params(discriminator))
    generator.train()
    discriminator.train()
    lr = 0.0002 if args['dcgan'] else 0.001
    betas = (0.5, 0.999) if args['dcgan'] else (0.0, 0.9)
    g_optim = Adam(trainable_params(generator), lr=lr, betas=betas)
    d_optim = Adam(trainable_params(discriminator), lr=lr * (4 if args['ttur'] else 1), betas=betas)
    train_loader = DataLoader(get_mnist_ds(64 if args['dcgan'] else 32), batch_size=args['batch_size'],
                              shuffle=True, drop_last=True)
    train_sampler = iter(infinite_sampler(train_loader))
    if args['moving_average']:
        smoothed_g_params = flatten_params(generator)
    if not args['no_fid']:
        fid_calculator = prepare_inception_metrics(64 if args['dcgan'] else 32)
    g_losses = []
    d_losses = []

    def get_z():
        return to(random_latents(args['batch_size'], args['z_dim'], args['z_distribution'])), \
               to(torch.randint(10, (args['batch_size'],)))

    def sample_generator():
        with torch.no_grad():
            return generator(*get_z())

    fixed_z = get_z()[0][:8 * 8]
    for idx in range(args['iterations']):
        current_d_losses = []
        for d_step in range(args['d_steps']):
            real, y_real = to(next(train_sampler))
            z, y_z = get_z()
            d_loss, fake = discriminator_loss(discriminator, generator, real, y_real, z, y_z, args['loss'],
                                              d_step == (args['d_steps'] - 1), args['iwass_drift_epsilon'],
                                              args['grad_lambda'], args['iwass_target'])
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()
            current_d_losses.append(d_loss.item())
        d_losses.append(sum(current_d_losses) / len(current_d_losses))
        if args['tensorboard']:
            writer.add_scalar('discriminator_loss', d_losses[-1], idx)
        current_g_losses = []
        for g_step in range(args['g_steps']):
            if g_step != 0:
                real, y_real = to(next(train_sampler))
                z, y_z = get_z()
            g_loss = generator_loss(discriminator, generator, real, y_real, z, y_z, args['loss'], g_step == 0, fake)
            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()
            current_g_losses.append(g_loss.item())
        g_losses.append(sum(current_g_losses) / len(current_g_losses))
        if args['tensorboard']:
            writer.add_scalar('generator_loss', g_losses[-1], idx)
        if args['moving_average']:
            update_flattened(generator, smoothed_g_params)
        if idx % args['eval_freq'] == 0:
            if args['moving_average']:
                original_g_params = flatten_params(generator)
                load_params(smoothed_g_params, generator)
            if not args['no_fid']:
                fid = fid_calculator(sample_generator)
                if args['tensorboard']:
                    writer.add_scalar('FID', fid, idx)
                if args['verbose']:
                    print(idx, 'fid', fid)
            torch.save({'g': generator.state_dict(), 'd': discriminator.state_dict()}, '{}.pth'.format(idx))
            if args['tensorboard']:
                with torch.no_grad():
                    writer.add_image('Fixed', vutils.make_grid(generator(fixed_z), range=(-1.0, 1.0), nrow=8), idx)
            if args['moving_average']:
                load_params(original_g_params, generator)
        if args['verbose'] and idx % 25 == 0:
            print(idx, 'g', g_losses[-1], 'd', d_losses[-1])
    if args['tensorboard']:
        writer.close()
    elif args['verbose']:
        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(g_losses, label="G")
        plt.plot(d_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()
        plt.figure(figsize=(15, 15))
        plt.subplot(1, 2, 1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(np.transpose(vutils.make_grid(next(train_sampler)[0][:64], range=(-1.0, 1.0), padding=5), (1, 2, 0)))
        plt.subplot(1, 2, 2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(
            np.transpose(vutils.make_grid(sample_generator()[:64].cpu(), range=(-1.0, 1.0), padding=5), (1, 2, 0)))
        plt.show()


def parse_args(args):
    parser = ArgumentParser()
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--bn_in_d', action='store_true',
                        help='whether to apply batch normalization in the discriminator of DCGAN or not')
    parser.add_argument('--no_fid', action='store_true')
    parser.add_argument('--cond', action='store_true', help='train a conditional gan(only works with SNGAN)')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--dcgan', action='store_true', help='use DCGAN or SNGAN')
    parser.add_argument('--z_dim', default=100, type=int)
    parser.add_argument('--experiment_name', default='', type=str)
    parser.add_argument('--model_dim', default=64, type=int)
    parser.add_argument('--no_spectral_norm', action='store_true', help='do not use sepctral norm in the discriminator')
    parser.add_argument('--ttur', action='store_true')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--d_steps', default=2, type=int)
    parser.add_argument('--g_steps', default=1, type=int)
    parser.add_argument('--loss', choices=['hinge', 'rsgan', 'rasgan', 'rahinge', 'wgan_gp', 'vanilla'],
                        default='hinge')
    parser.add_argument('--grad_lambda', default=0.0, type=float)  # used to be 10.0
    parser.add_argument('--iwass_drift_epsilon', default=0.001, type=float)
    parser.add_argument('--iwass_target', default=1.0, type=float)
    parser.add_argument('--z_distribution', choices=['normal', 'bernoulli', 'censored', 'uniform'], default='normal')
    parser.add_argument('--iterations', default=4501, type=int)
    parser.add_argument('--eval_freq', default=250, type=int)
    parser.add_argument('--moving_average', action='store_true', help='use a moving average of G for evaluation')
    parser.add_argument('--tensorboard', action='store_true')
    return vars(parser.parse_args(args))


if __name__ == '__main__':
    best_args = '--verbose --eval_freq 250 --moving_average --tensorboard'.split()
    train_gan(parse_args(args=best_args))
