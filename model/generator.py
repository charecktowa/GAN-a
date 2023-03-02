import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, ngpu: int, z_dim: int, in_channels: int, ngf: int) -> None:
        super(Generator, self).__init__()

        self.ngpu = ngpu

        self.main = nn.Sequential(
            self._block(z_dim, ngf * 8, 4, 1, 0),
            self._block(ngf * 8, ngf * 4, 4, 2, 1),
            self._block(ngf * 4, ngf * 2, 4, 2, 1),
            self._block(ngf * 2, ngf * 4, 4, 2, 1),
            nn.ConvTranspose2d(ngf, in_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def _block(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
    ) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)
