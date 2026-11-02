import torch
import torch.nn as nn

class DoubleConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.GELU()
        )
        
    def forward(self, x):
        return self.double_conv(x)

class DecoupledBasicBlock1D(nn.Module):
    expansion = 1 
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, dropout_p=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_p)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * self.expansion)
            )
        nn.init.constant_(self.bn2.weight, 0)

    def forward(self, x):
        identity = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.act(out)

class UpgradedResBlock1D(nn.Module):
    def __init__(self, channels, dilation=1, dropout_p=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        nn.init.constant_(self.bn2.weight, 0)
    def forward(self, x):
        res = x
        x = self.act(self.bn1(self.conv1(x)))
        x = self.dropout(x)
        x = self.bn2(self.conv2(x))
        return self.act(x + res)
