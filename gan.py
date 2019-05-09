from torch.optim import Adam
from argparse import ArgumentParser
from torch.utils.data import DataLoader

from losses import discriminator_loss, generator_loss
from modules import DCGenerator, DCDiscriminator, ResNetGenerator, ResNetDiscriminator
from utils import to, num_params, trainable_params, get_mnist_ds, enable_benchmark, infinite_sampler, random_latents


def train_gan(args):
    enable_benchmark()
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
    for idx in range(args['iterations']):
        for _ in range(args['d_steps']):
            real = next(train_sampler)
            z = random_latents(args['batch_size'], args['z_dim'], args['z_distribution'])
            d_loss = discriminator_loss(discriminator, generator, real, z, args['loss'],
                                        args['iwass_drift_epsilon'], args['grad_lambda'], args['iwass_target'])
            d_optim.zero_grad()
            d_loss.backward()
            d_optim.step()
        for _ in range(args['g_steps']):
            real = next(train_sampler)
            z = random_latents(args['batch_size'], args['z_dim'], args['z_distribution'])
            g_loss = generator_loss(discriminator, generator, real, z, args['loss'])
            g_optim.zero_grad()
            g_loss.backward()
            g_optim.step()
        # moving average of G, monitor g_loss and d_loss, fid eval + save, fixed random sample
        # reverse, reverse as attack detector, fine-tune discriminator, disc as attack detector, multi-gen reverse
        # vanilla multi autoencoder, train classifier, attacks!()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('verbose', action='store_true')
    parser.add_argument('dcgan', action='store_true')
    parser.add_argument('z_dim', default=100, type=int)
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
    return parser.parse_args()
