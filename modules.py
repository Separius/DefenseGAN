import numpy as np
from torch import nn
from torch.nn.utils import spectral_norm


# translated from https://github.com/pfnet-research/sngan_projection/blob/master/gen_models/resblocks.py
class ResBlockGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, apply_sn=False):
        super().__init__()
        conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        bypass_conv = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.xavier_uniform_(conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(conv2.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(bypass_conv.weight.data, 1.0)
        if apply_sn:
            conv1 = spectral_norm(conv1)
            conv2 = spectral_norm(conv2)
            bypass_conv = spectral_norm(bypass_conv)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            conv2
        )
        self.bypass = nn.Sequential(nn.Upsample(scale_factor=2), bypass_conv)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


# translated from https://github.com/pfnet-research/sngan_projection/blob/master/gen_models/resnet.py
class ResNetGenerator(nn.Module):
    def __init__(self, z_dim=100, rgb_channels=1, dim=64, apply_sn=False):
        super().__init__()
        self.z_dim = z_dim
        self.dense = nn.Linear(z_dim, 2 * 2 * dim * 8)
        final = nn.Conv2d(dim, rgb_channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(final.weight.data, 1.)
        if apply_sn:
            self.dense = spectral_norm(self.dense)
            final = spectral_norm(final)

        self.model = nn.Sequential(
            ResBlockGenerator(dim * 8, dim * 4, apply_sn),
            ResBlockGenerator(dim * 4, dim * 2, apply_sn),
            ResBlockGenerator(dim * 2, dim * 1, apply_sn),
            ResBlockGenerator(dim * 1, dim * 1, apply_sn),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            final,
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(self.dense(z).view(z.size(0), -1, 2, 2))


# translated from https://github.com/pfnet-research/sngan_projection/blob/master/dis_models/resblocks.py
class ResBlockDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, apply_sn=False, is_first_layer=False):
        super().__init__()
        conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        bypass_conv = nn.Conv2d(in_channels, out_channels, 1)
        nn.init.xavier_uniform_(conv1.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(conv2.weight.data, np.sqrt(2))
        nn.init.xavier_uniform_(bypass_conv.weight.data, 1.0)
        if apply_sn:
            conv1 = spectral_norm(conv1)
            conv2 = spectral_norm(conv2)
            bypass_conv = spectral_norm(bypass_conv)

        self.model = nn.Sequential(
            nn.Identity() if is_first_layer else nn.ReLU(),
            conv1,
            nn.ReLU(),
            conv2,
            nn.AvgPool2d(2)
        )
        self.bypass = nn.Sequential(bypass_conv, nn.AvgPool2d(2))

    def forward(self, x):
        return self.model(x) + self.bypass(x)


# translated from https://github.com/pfnet-research/sngan_projection/blob/master/dis_models/snresnet.py
class ResNetDiscriminator(nn.Module):
    def __init__(self, rgb_channels=1, dim=64, apply_sn=False):
        super().__init__()
        self.model = nn.Sequential(
            ResBlockDiscriminator(rgb_channels, dim, apply_sn, is_first_layer=True),
            ResBlockDiscriminator(dim, dim * 2, apply_sn),
            ResBlockDiscriminator(dim * 2, dim * 4, apply_sn),
            ResBlockDiscriminator(dim * 4, dim * 8, apply_sn),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(dim * 8, 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        if apply_sn:
            self.fc = spectral_norm(self.fc)

    def forward(self, x):
        return self.fc(self.model(x).view(x.size(0), -1)).squeeze(dim=1)


def dcgan_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# borrowed from https://github.com/pytorch/tutorials/blob/master/beginner_source/dcgan_faces_tutorial.py
class DCGenerator(nn.Module):
    def __init__(self, z_dim=100, rgb_channels=1, dim=64):
        super().__init__()
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(z_dim, dim * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(dim * 8),
            nn.ReLU(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(dim * 8, dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim * 4),
            nn.ReLU(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(dim * 4, dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim * 2),
            nn.ReLU(),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(dim * 2, dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(dim, rgb_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, z):
        return self.model(z.view(z.size(0), -1, 1, 1))


# borrowed from https://github.com/pytorch/examples/blob/master/dcgan/main.py
class DCDiscriminator(nn.Module):
    def __init__(self, rgb_channels=1, dim=64):
        super().__init__()
        self.model = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(rgb_channels, dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(dim, dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim * 2),
            nn.LeakyReLU(0.2),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(dim * 2, dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim * 4),
            nn.LeakyReLU(0.2),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(dim * 4, dim * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim * 8),
            nn.LeakyReLU(0.2),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(dim * 8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, x):
        return self.model(x).squeeze()
