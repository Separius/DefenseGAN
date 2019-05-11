import numpy as np
from torch import nn
from torch.nn.utils import spectral_norm


class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_classes):
        super().__init__()
        self.num_features = num_features
        if num_classes < 2:
            self.bn = nn.BatchNorm2d(num_features, affine=True)
            self.embed = None
        else:
            self.bn = nn.BatchNorm2d(num_features, affine=False)
            self.embed = nn.Embedding(num_classes, num_features * 2)
            self.embed.weight.data[:, :num_features].normal_(1, 0.02)  # Initialise scale at N(1, 0.02)
            self.embed.weight.data[:, num_features:].zero_()  # Initialise bias at 0

    def forward(self, x, y):
        out = self.bn(x)
        if self.embed is None:
            return out
        gamma, beta = self.embed(y).chunk(2, 1)
        out = gamma.view(-1, self.num_features, 1, 1) * out + beta.view(-1, self.num_features, 1, 1)
        return out


# translated from https://github.com/pfnet-research/sngan_projection/blob/master/gen_models/resblocks.py
class ResBlockGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, apply_sn=False, num_classes=-1):
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
        self.bn1 = ConditionalBatchNorm2d(in_channels, num_classes)
        self.cnn1 = nn.Sequential(
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            conv1,
        )
        self.bn2 = ConditionalBatchNorm2d(out_channels, num_classes)
        self.cnn2 = nn.Sequential(
            nn.ReLU(),
            conv2
        )
        self.bypass = nn.Sequential(nn.Upsample(scale_factor=2), bypass_conv)

    def forward(self, x, y=None):
        return self.cnn2(self.bn2(self.cnn1(self.bn1(x, y)), y)) + self.bypass(x)


# translated from https://github.com/pfnet-research/sngan_projection/blob/master/gen_models/resnet.py
class ResNetGenerator(nn.Module):
    def __init__(self, z_dim=100, rgb_channels=1, dim=64, apply_sn=False, num_classes=-1):
        super().__init__()
        self.z_dim = z_dim
        self.dense = nn.Linear(z_dim, 2 * 2 * dim * 8)
        final = nn.Conv2d(dim, rgb_channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(final.weight.data, 1.)
        if apply_sn:
            self.dense = spectral_norm(self.dense)
            final = spectral_norm(final)
        self.blocks = nn.ModuleList([
            ResBlockGenerator(dim * 8, dim * 4, apply_sn, num_classes),
            ResBlockGenerator(dim * 4, dim * 2, apply_sn, num_classes),
            ResBlockGenerator(dim * 2, dim, apply_sn, num_classes),
            ResBlockGenerator(dim, dim, apply_sn, num_classes)
        ])
        self.model = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            final,
            nn.Tanh()
        )

    def forward(self, z, y=None):
        x = self.dense(z).view(z.size(0), -1, 2, 2)
        for b in self.blocks:
            x = b(x, y)
        return self.model(x)


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
    def __init__(self, rgb_channels=1, dim=64, apply_sn=False, apply_bn=None, num_classes=-1):
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
        if num_classes > 1:
            self.embedding = nn.Embedding(num_classes, dim * 8)
            nn.init.xavier_uniform_(self.embedding.weight.data, 1.)
            if apply_sn:
                self.embedding = spectral_norm(self.embedding)
        else:
            self.embedding = None

    def forward(self, x, y=None):
        h = self.model(x).view(x.size(0), -1)
        o = self.fc(h).squeeze(dim=1)
        if self.embedding is None:
            return o
        return o + (self.embedding(y) * h).sum(dim=1)


# borrowed from https://github.com/pytorch/tutorials/blob/master/beginner_source/dcgan_faces_tutorial.py
class DCGenerator(nn.Module):
    def __init__(self, z_dim=100, rgb_channels=1, dim=64, apply_sn=False, num_classes=-1):
        assert num_classes == -1
        super().__init__()
        self.apply_sn = apply_sn
        self.z_dim = z_dim
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            *self.get_conv(z_dim, dim * 8, 1, 0),
            # state size. (ngf*8) x 4 x 4
            *self.get_conv(dim * 8, dim * 4, 2, 1),
            # state size. (ngf*4) x 8 x 8
            *self.get_conv(dim * 4, dim * 2, 2, 1),
            # state size. (ngf*2) x 16 x 16
            *self.get_conv(dim * 2, dim, 2, 1),
            # state size. (ngf) x 32 x 32
            *self.get_conv(dim, rgb_channels, 2, 1, last=True),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def get_conv(self, in_channels, out_channels, stride, padding, last=False):
        conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4,
                                  stride=stride, padding=padding, bias=False)
        nn.init.normal_(conv.weight.data, 0.0, 0.02)
        if self.apply_sn:
            conv = spectral_norm(conv)
        if last:
            return conv,
        bn = nn.BatchNorm2d(out_channels)
        nn.init.normal_(bn.weight.data, 1.0, 0.02)
        nn.init.constant_(bn.bias.data, 0)
        return conv, bn, nn.ReLU()

    def forward(self, z):
        return self.model(z.view(z.size(0), -1, 1, 1))


# borrowed from https://github.com/pytorch/examples/blob/master/dcgan/main.py
class DCDiscriminator(nn.Module):
    def __init__(self, rgb_channels=1, dim=64, apply_sn=False, apply_bn=True, num_classes=-1):
        assert num_classes == -1
        super().__init__()
        self.apply_sn = apply_sn
        self.apply_bn = apply_bn
        self.model = nn.Sequential(
            # input is (nc) x 64 x 64
            *self.get_conv(rgb_channels, dim, 2, 1, False),
            nn.LeakyReLU(0.2),
            # state size. (ndf) x 32 x 32
            *self.get_conv(dim, dim * 2, 2, 1, True),
            # state size. (ndf*2) x 16 x 16
            *self.get_conv(dim * 2, dim * 4, 2, 1, True),
            # state size. (ndf*4) x 8 x 8
            *self.get_conv(dim * 4, dim * 8, 2, 1, True),
            # state size. (ndf*8) x 4 x 4
            *self.get_conv(dim * 8, 1, 1, 0, False)
        )

    def get_conv(self, in_channels, out_channels, stride, padding, middle):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=4,
                         stride=stride, padding=padding, bias=not self.apply_bn)
        nn.init.normal_(conv.weight.data, 0.0, 0.02)
        if self.apply_sn:
            conv = spectral_norm(conv)
        if middle and self.apply_bn:
            bn = nn.BatchNorm2d(out_channels)
            nn.init.normal_(bn.weight.data, 1.0, 0.02)
            nn.init.constant_(bn.bias.data, 0)
            return conv, bn, nn.LeakyReLU(0.2)
        return conv,

    def forward(self, x):
        return self.model(x).squeeze()


class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(1, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(16, 32, 5), nn.ReLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(32, 64, 3), nn.ReLU())
        self.fc = nn.Sequential(nn.Linear(3 * 3 * 64, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        x = self.cnn(x)
        return self.fc(x.view(x.size(0), -1))


class MLPClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        input_size = 32 * 32
        self.fc = nn.Sequential(nn.Linear(input_size, 512), nn.ReLU(), nn.BatchNorm1d(512),
                                nn.Linear(512, 128), nn.ReLU(), nn.BatchNorm1d(128),
                                nn.Linear(128, 10))

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


class CNNAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8, 8, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 3, stride=2, output_padding=1, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 3, stride=2, output_padding=1, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(8, 8, 3, stride=2, output_padding=1, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 3, stride=2, output_padding=1, padding=1), nn.Tanh(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class MLPAutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 32 * 32),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x).view(x.size(0), -1)).view(x.size(0), 1, 32, 32)
