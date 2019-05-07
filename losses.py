import torch
import torch.nn.functional as F
from torch.autograd import Variable, grad

one = None
zero = None
mixing_factors = None


def get_mixing_factor(batch_size, device):
    global mixing_factors
    if mixing_factors is None or batch_size != mixing_factors.size(0):
        mixing_factors = torch.FloatTensor(batch_size, 1, 1).to(device)
    mixing_factors.uniform_()
    return mixing_factors


def get_one(batch_size, device):
    global one
    if one is None or batch_size != one.size(0):
        one = torch.ones(batch_size).to(device)
    return one


def get_zero(batch_size, device):
    global zero
    if zero is None or batch_size != zero.size(0):
        zero = torch.zeros(batch_size).to(device)
    return zero


def calc_grad(x_hat, pred_hat):
    return grad(outputs=pred_hat, inputs=x_hat, grad_outputs=get_one(pred_hat.size(0), pred_hat),
                create_graph=True, retain_graph=True, only_inputs=True)[0]


def generator_loss(dis, gen, real, z, loss_type: str):
    gen.zero_grad()
    g_ = gen(z)
    d_fake = dis(g_)
    if loss_type in {'hinge', 'wgan_gp'}:
        g_loss = -d_fake.mean()
    else:
        with torch.no_grad():
            d_real = dis(real)
        if loss_type == 'rsgan':
            g_loss = F.binary_cross_entropy_with_logits(d_fake - d_real, get_one(d_fake.size(0), d_fake))
        elif loss_type == 'rasgan':
            batch_size = d_fake.size(0)
            g_loss = (F.binary_cross_entropy_with_logits(d_fake - d_real.mean(), get_one(batch_size, d_fake)) +
                      F.binary_cross_entropy_with_logits(d_real - d_fake.mean(), get_zero(batch_size, z))) / 2.0
        elif loss_type == 'rahinge':
            g_loss = (torch.mean(F.relu(1.0 + (d_real - torch.mean(d_fake)))) + torch.mean(
                F.relu(1.0 - (d_fake - torch.mean(d_real))))) / 2
        else:
            raise ValueError('Invalid loss type')
    return g_loss


def discriminator_loss(dis: torch.nn.Module, gen: torch.nn.Module, real, z, loss_type: str,
                       iwass_drift_epsilon: float, grad_lambda: float, iwass_target: float):
    dis.zero_grad()
    d_real = dis(real)
    with torch.no_grad():
        g_ = gen(z)
    d_fake = dis(g_)
    batch_size = d_real.size(0)
    if loss_type == 'hinge':
        d_loss = F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()
    elif loss_type == 'rsgan':
        d_loss = F.binary_cross_entropy_with_logits(d_real - d_fake, get_one(batch_size, d_fake))
    elif loss_type == 'rasgan':
        d_loss = (F.binary_cross_entropy_with_logits(d_real - d_fake.mean(), get_one(batch_size, d_fake)) +
                  F.binary_cross_entropy_with_logits(d_fake - d_real.mean(), get_zero(batch_size, z))) / 2.0
    elif loss_type == 'rahinge':
        d_loss = (torch.mean(F.relu(1.0 - (d_real - d_fake.mean()))) + torch.mean(
            F.relu(1.0 + (d_fake - d_real.mean())))) / 2
    elif loss_type == 'wgan_gp':
        d_fake_mean = d_fake.mean()
        d_real_mean = d_real.mean()
        d_loss = d_fake_mean - d_real_mean + (d_real ** 2).mean() * iwass_drift_epsilon
    else:
        raise ValueError('Invalid loss type')
    if grad_lambda != 0:
        alpha = get_mixing_factor(real.size(0), z)
        x_hat = Variable(alpha * real.data + (1.0 - alpha) * g_.data, requires_grad=True)
        pred_hat = dis(x_hat)
        g = calc_grad(x_hat, pred_hat).view(batch_size, -1)
        gp = g.norm(p=2, dim=1) - iwass_target
        gp_loss = (gp ** 2).mean() * grad_lambda / (iwass_target ** 2)
        d_loss = d_loss + gp_loss
    return d_loss
