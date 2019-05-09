import torch
import torch.nn.functional as F
from torch.autograd import Variable, grad
from torch.nn.functional import binary_cross_entropy_with_logits as bce

one = None
zero = None
mixing_factors = None


def get_mixing_factor(x):
    global mixing_factors
    if mixing_factors is None or x.size(0) != mixing_factors.size(0):
        mixing_factors = torch.FloatTensor(x.size(0), 1, 1).to(x)
    mixing_factors.uniform_()
    return mixing_factors


def get_one(x):
    global one
    if one is None or x.size(0) != one.size(0):
        one = torch.ones(x.size(0)).to(x)
    return one


def get_zero(x):
    global zero
    if zero is None or x.size(0) != zero.size(0):
        zero = torch.zeros(x.size(0)).to(x)
    return zero


def calc_grad(x_hat, pred_hat):
    return grad(outputs=pred_hat, inputs=x_hat, grad_outputs=get_one(pred_hat),
                create_graph=True, retain_graph=True, only_inputs=True)[0]


def generator_loss(dis, gen, real, z, loss_type: str):
    gen.zero_grad()
    g_ = gen(z)
    d_fake = dis(g_)
    if loss_type in {'hinge', 'wgan_gp'}:
        return -d_fake.mean()
    with torch.no_grad():
        d_real = dis(real)
    if loss_type == 'rsgan':
        return bce(d_fake - d_real, get_one(d_fake))
    elif loss_type == 'rasgan':
        return (bce(d_fake - d_real.mean(), get_one(d_fake)) +
                bce(d_real - d_fake.mean(), get_zero(d_real))) / 2.0
    elif loss_type == 'rahinge':
        return (F.relu(1.0 + (d_real - d_fake.mean())).mean() +
                F.relu(1.0 - (d_fake - d_real.mean())).mean()) / 2
    else:
        raise ValueError('Invalid loss type')


def discriminator_loss(dis: torch.nn.Module, gen: torch.nn.Module, real, z, loss_type: str,
                       iwass_drift_epsilon: float, grad_lambda: float, iwass_target: float):
    dis.zero_grad()
    d_real = dis(real)
    with torch.no_grad():
        g_ = gen(z)
    d_fake = dis(g_)
    if loss_type == 'hinge':
        d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
    elif loss_type == 'rsgan':
        d_loss = bce(d_real - d_fake, get_one(d_fake))
    elif loss_type == 'rasgan':
        d_loss = (bce(d_real - d_fake.mean(), get_one(d_fake)) +
                  bce(d_fake - d_real.mean(), get_zero(z))) / 2.0
    elif loss_type == 'rahinge':
        d_loss = (F.relu(1.0 - (d_real - d_fake.mean())).mean() +
                  F.relu(1.0 + (d_fake - d_real.mean())).mean()) / 2
    elif loss_type == 'wgan_gp':
        d_loss = d_fake.mean() - d_real.mean() + (d_real ** 2).mean() * iwass_drift_epsilon
    else:
        raise ValueError('Invalid loss type')
    if grad_lambda != 0:
        alpha = get_mixing_factor(real)
        x_hat = Variable(alpha * real.data + (1.0 - alpha) * g_.data, requires_grad=True)
        g = calc_grad(x_hat, dis(x_hat)).view(x_hat.size(0), -1)
        gp = g.norm(p=2, dim=1) - iwass_target
        gp_loss = (gp ** 2).mean() * grad_lambda / (iwass_target ** 2)
        d_loss = d_loss + gp_loss
    return d_loss
