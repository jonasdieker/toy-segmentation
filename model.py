import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, features: int = 64):
        super().__init__()

        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 3, padding=1), nn.ReLU()
        )

        self.down1 = Block(features, features * 2, down=True)
        self.down2 = Block(features * 2, features * 4, down=True)
        self.down3 = Block(features * 4, features * 8, down=True)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 16, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(features * 16, features * 8, 3, padding=1),
            nn.ReLU(),
        )

        self.up1 = Block(features * 8 * 2, features * 4, down=False)
        self.up2 = Block(features * 4 * 2, features * 2, down=False)
        self.up3 = Block(features * 2 * 2, features, down=False)

        self.final_up = nn.Sequential(nn.ConvTranspose2d(features * 2, out_channels, 1))

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        bottleneck = self.bottleneck(d4)
        up1 = self.up1(torch.cat([bottleneck, d4], 1))
        up2 = self.up2(torch.cat([up1, d3], 1))
        up3 = self.up3(torch.cat([up2, d2], 1))
        out = self.final_up(torch.cat([up3, d1], 1))
        return out


if __name__ == "__main__":
    net = UNet(3, 10)
    input = torch.randn((1, 3, 256, 256))
    print(f"input shape: {input.shape}")
    print(f"output shape: {net(input).shape}")
