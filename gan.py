import torch
from torch.optim import Adam
import torchvision.utils as vutils
from argparse import ArgumentParser
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from fid import prepare_inception_metrics
from losses import discriminator_loss, generator_loss
from modules import DCGenerator, DCDiscriminator, ResNetGenerator, ResNetDiscriminator
from utils import (to, num_params, trainable_params, get_mnist_ds, enable_benchmark,
                   infinite_sampler, random_latents, flatten_params, update_flattened, load_params)


def reconstruct(gen, x, args):
    z = random_latents(x.size(0) * args['recon_restarts'], args['z_dim'], args['z_distribution'])
    z = torch.nn.Parameter(z, requires_grad=True)
    x = x.repeat(args['recon_restarts'], dim=0)
    for _ in range(args['recon_iters']):
        loss = ((x - gen(z)) ** 2).mean(dim=[1, 2, 3])
        z.zero_grad()
        loss.mean().backward()
        z.add_(-z.grad * args['recon_step_size'])
    return gen(z), loss


def train_gan(args):
    enable_benchmark()
    writer = SummaryWriter(args['exp_name'] if args['exp_name'] != '' else None)
    common_nn_args = dict(rgb_channels=1, dim=args['model_dim'], apply_sn=not args['no_spectral_norm'])
    if args['dcgan']:
        generator = DCGenerator
        discriminator = DCDiscriminator
    else:
        generator = ResNetGenerator
        discriminator = ResNetDiscriminator
    generator = to(generator(**common_nn_args, z_dim=args['z_dim']))
    discriminator = to(discriminator(**common_nn_args))
    if args['verbose']:
        print('Generator:')
        print(generator)
        print('num params:', num_params(generator))
        print('\nDiscriminator:')
        print(discriminator)
        print('num params:', num_params(discriminator))
    g_optim = Adam(trainable_params(generator), lr=0.0002, betas=(0.5, 0.999) if args['dcgan'] else (0.0, 0.99))
    d_optim = Adam(trainable_params(generator), lr=0.0002 * (4 if args['ttur'] else 1),
                   betas=(0.5, 0.999) if args['dcgan'] else (0.0, 0.99))
    train_loader = DataLoader(get_mnist_ds(), batch_size=args['batch_size'], shuffle=True, drop_last=True)
    train_sampler = iter(infinite_sampler(train_loader))
    smoothed_g_params = flatten_params(generator)
    fid_calculator = prepare_inception_metrics()
    g_losses = []
    d_losses = []

    def get_z():
        return random_latents(args['batch_size'], args['z_dim'], args['z_distribution'])

    def sample_generator():
        with torch.no_grad():
            return generator(get_z())

    fixed_z = get_z()[:8 * 8]
    for idx in range(args['iterations']):
        current_d_losses = []
        for _ in range(args['d_steps']):
            real, _ = next(train_sampler)
            z = get_z()
            d_loss = discriminator_loss(discriminator, generator, real, z, args['loss'],
                                        args['iwass_drift_epsilon'], args['grad_lambda'], args['iwass_target'])
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()
            current_d_losses.append(d_loss.item())
        d_losses.append(sum(current_d_losses) / len(current_d_losses))
        writer.add_scalar('discriminator_loss', d_losses[-1], idx)
        current_g_losses = []
        for _ in range(args['g_steps']):
            real, _ = next(train_sampler)
            z = get_z()
            g_loss = generator_loss(discriminator, generator, real, z, args['loss'])
            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()
            current_g_losses.append(g_loss.item())
        g_losses.append(sum(current_g_losses) / len(current_g_losses))
        writer.add_scalar('generator_loss', g_losses[-1], idx)
        update_flattened(generator, smoothed_g_params)
        if idx % args['eval_freq'] == 0:
            original_g_params = flatten_params(generator)
            load_params(smoothed_g_params, generator)
            fid = fid_calculator(sample_generator)
            writer.add_scalar('FID', fid, idx)
            if args['verbose']:
                print('fid', fid)
            torch.save({'g': generator.state_dict(), 'd': discriminator.state_dict()}, '{}.pth'.format(idx))
            x = next(train_sampler)[:8]
            x_r, _ = reconstruct(generator, x, args)
            writer.add_image('Real', vutils.make_grid(x, range=(-1.0, 1.0), nrow=8), idx)
            writer.add_image('Recon', vutils.make_grid(x_r, range=(-1.0, 1.0), nrow=8), idx)
            with torch.no_grad():
                writer.add_image('Fixed', vutils.make_grid(generator(fixed_z), range=(-1.0, 1.0), nrow=8), idx)
            load_params(original_g_params, generator)
        if args['verbose']:
            print(idx, 'g', g_losses[-1], 'd', d_losses[-1])
        # reverse, reverse as attack detector, fine-tune discriminator,
        # disc as attack detector, multi-gen reverse, vanilla multi autoencoder
        # train classifier, attacks!() + attack samples + reverse samples + adv training
        # easy defences(binary, jpeg, mean, median)
    writer.close()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('verbose', action='store_true')
    parser.add_argument('dcgan', action='store_true')
    parser.add_argument('z_dim', default=100, type=int)
    parser.add_argument('exp_name', default='', type=str)
    parser.add_argument('model_dim', default=64, type=int)
    parser.add_argument('no_spectral_norm', action='store_true')
    parser.add_argument('ttur', action='store_true')
    parser.add_argument('batch_size', default=128, type=int)
    parser.add_argument('d_steps', default=5, type=int)
    parser.add_argument('g_steps', default=1, type=int)
    parser.add_argument('loss', choices=['hinge', 'rsgan', 'rasgan', 'rahinge', 'wgan_gp'], default='wgan_gp')
    parser.add_argument('grad_lambda', default=10.0, type=float)
    parser.add_argument('iwass_drift_epsilon', default=0.001, type=float)
    parser.add_argument('iwass_target', default=1.0, type=float)
    parser.add_argument('z_distribution', choices=['normal', 'bernoulli', 'censored', 'uniform'], default='normal')
    parser.add_argument('iterations', default=50000, type=int)
    parser.add_argument('eval_freq', default=1000, type=int)
    parser.add_argument('recon_restarts', default=8, type=int)
    parser.add_argument('recon_iters', default=200, type=int)
    parser.add_argument('recon_step_size', default=0.001, type=float)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_gan(args)
