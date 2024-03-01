import torch
import torch.nn as nn


class convbnrelu(nn.Module):

    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)


class DSConv3x3(nn.Module):

    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel), convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu))

    def forward(self, x):
        return self.conv(x)


class ConvOut(nn.Module):

    def __init__(self, in_channel):
        super(ConvOut, self).__init__()
        self.conv = nn.Sequential(nn.Dropout2d(p=0.1), nn.Conv2d(in_channel, 1, 1, stride=1, padding=0), nn.Sigmoid())

    def forward(self, x):
        return self.conv(x)


class MI_Module(nn.Module):  # Multiscale Interactive Module

    def __init__(self, channel):
        super(MI_Module, self).__init__()
        # Parameters Setting
        self.channel = channel

        # Module layer
        self.Dconv = nn.ModuleList([convbnrelu(channel, channel, k=3, s=1, p=2**i, d=2**i, g=channel) for i in range(4)])

        self.Pconv = nn.ModuleList([convbnrelu(4, 1, k=1, s=1, p=0) for i in range(channel)])

        self.Pfusion = convbnrelu(channel, channel, k=1, s=1, p=0)

        self.relu = nn.ReLU(inplace=True)

    def Shuffle(self, Fea_list, channel):
        MsFea_sp = [torch.chunk(x, channel, dim=1) for x in Fea_list]  # Multi-Scale Feature(Spilit)
        IrFea = [torch.cat((MsFea_sp[0][i], MsFea_sp[1][i], MsFea_sp[2][i], MsFea_sp[3][i]), 1) for i in range(channel)]  # Interactive Feature

        return IrFea

    def forward(self, x):
        MsFea = [conv(x) for conv in self.Dconv]  # Multi-Scale Feature

        IrFea = self.Shuffle(MsFea, self.channel)
        IrFea = [self.Pconv[i](IrFea[i]) for i in range(self.channel)]  # Interactive Feature

        IrFea = [torch.squeeze(x, dim=1) for x in IrFea]
        out = self.Pfusion(torch.stack(IrFea, 1))

        out = self.relu(out + x)

        return out
