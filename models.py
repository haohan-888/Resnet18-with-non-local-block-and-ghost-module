import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.theta = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.phi = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.g = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.out = nn.Conv2d(in_channels // 2, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        if x.size(2) == 1: return x

        B, C, H, W = x.shape
        theta = self.theta(x).view(B, -1, H * W)
        phi = self.phi(x).view(B, -1, H * W).permute(0, 2, 1)
        g = self.g(x).view(B, -1, H * W)

        attn = F.softmax(torch.bmm(theta, phi), dim=-1)
        out = torch.bmm(g, attn.permute(0, 2, 1)).view(B, -1, H, W)
        return x + self.gamma * self.out(out)


class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, ratio=2):
        super().__init__()
        hidden = out_channels // ratio
        self.primary = nn.Conv2d(in_channels, hidden, 1)
        self.cheap = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.conv = nn.Conv2d(hidden * 2, out_channels, 1)

    def forward(self, x):
        x1 = self.primary(x)
        x2 = self.cheap(x1)
        return self.conv(torch.cat([x1, x2], dim=1))


class AutoDriveNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.resnet.maxpool = nn.Identity()

        self.non_local = NonLocalBlock(512)
        self.ghost = GhostModule(512, 512)

        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Tanh()
        )
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        x = self.resnet(x).view(-1, 512, 1, 1)
        nl_out = self.non_local(x)
        gh_out = self.ghost(x)
        x = 0.6 * nl_out + 0.4 * gh_out
        return self.head(x.flatten(1))