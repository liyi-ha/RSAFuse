import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class RegFuseUNet(nn.Module):
    def __init__(self, in_channels=8, num_classes=5, base_channels=32):
        super().__init__()
        c = base_channels
        self.inc = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 12)

        self.up1 = Up(c * 12, c * 8, c * 8)
        self.up2 = Up(c * 8, c * 4, c * 4)
        self.up3 = Up(c * 4, c * 2, c * 2)
        self.up4 = Up(c * 2, c, c)

        self.seg_head = nn.Conv2d(c, num_classes, kernel_size=1)
        self.alpha_head = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.detail_gate_head = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.residual_head = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 1, kernel_size=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.seg_head(x), self.alpha_head(x), self.detail_gate_head(x), self.residual_head(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
