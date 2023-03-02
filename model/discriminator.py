import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, ngpu: int, in_channels: int, ndf: int) -> None:
        super(Discriminator, self).__init__()

        self.ngpu = ngpu

        self.main = nn.Sequential(
            # Input is (in_channels) x 64 x 64
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            # 32x32
            nn.LeakyReLU(0.2, inplace=True),
            self._block(ndf, ndf * 2, 4, 2, 1),  # 16x16
            self._block(ndf * 2, ndf * 4, 4, 2, 1),  # 8x8
            self._block(ndf * 4, ndf * 8, 4, 2, 1),  # 4x4
            nn.Conv2d(
                ndf * 8, 1, kernel_size=4, stride=2, padding=0, bias=False
            ),  # 1x1
            nn.Sigmoid(),
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
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x)
