from torch import nn
from torch.nn.utils import spectral_norm


class DCGenerator(nn.Module):
    def __init__(self, z_dim, rgb_channels=1, apply_sn=False):
        super().__init__()
        self.z_dim = z_dim
        self.apply_sn = apply_sn
        act = nn.ReLU()
        self.model = nn.Sequential(
            self.get_conv(z_dim, 512, 4, stride=1),
            nn.BatchNorm2d(512),
            act,
            self.get_conv(512, 256, 4, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(256),
            act,
            self.get_conv(256, 128, 4, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(128),
            act,
            self.get_conv(128, 64, 4, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(64),
            act,
            self.get_conv(64, rgb_channels, 3, stride=1, padding=(1, 1)),
            nn.Tanh(),
        )

    def get_conv(self, *args, **kwargs):
        conv = nn.ConvTranspose2d(*args, **kwargs)
        if self.apply_sn:
            return spectral_norm(conv)
        return conv

    def forward(self, z):
        return self.model(z.view(-1, self.z_dim, 1, 1))


class DCDiscriminator(nn.Module):
    def __init__(self, rgb_channels=1, apply_sn=False):
        super().__init__()
        self.apply_sn = apply_sn
        act = nn.LeakyReLU(0.2)
        self.model = nn.Sequential(
            self.get_conv(rgb_channels, 64, 3, stride=1, padding=(1, 1)),
            act,
            self.get_conv(64, 64, 4, stride=2, padding=(1, 1)),
            act,
            self.get_conv(64, 128, 3, stride=1, padding=(1, 1)),
            act,
            self.get_conv(128, 128, 4, stride=2, padding=(1, 1)),
            act,
            self.get_conv(128, 256, 3, stride=1, padding=(1, 1)),
            act,
            self.get_conv(256, 256, 4, stride=2, padding=(1, 1)),
            act,
            self.get_conv(256, 512, 3, stride=1, padding=(1, 1)),
            act,
        )
        self.fc = nn.Linear(4 * 4 * 512, 1)
        if self.apply_sn:
            self.fc = spectral_norm(self.fc)

    def get_conv(self, *args, **kwargs):
        conv = nn.Conv2d(*args, **kwargs)
        if self.apply_sn:
            return spectral_norm(conv)
        return conv

    def forward(self, x):
        h = self.model(x).view(x.size(0), -1)
        return self.fc(h)
