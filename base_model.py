import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def generate_it(x, t=0, nf=3, f=7):
    index = np.array([t - nf // 2 + i for i in range(nf)])
    index = np.clip(index, 0, f-1).tolist()
    it = x[:, index, :, :]
    return it

def make_layer(block, num_of_layer):
    layers = []
    for _ in range(num_of_layer):
        layers.append(block())
    return nn.Sequential(*layers)

class ConBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        return out


class ChannelAttention(nn.Module):
    def __init__(self, num_features, reduction):
        super(ChannelAttention, self).__init__()
        self.module = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_features, num_features // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.module(x)

class SKFF(nn.Module):
    def __init__(self, in_channels, height=2, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]
        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)
        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V



class SKU_Net(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.nf = nf
        base_ks = 3
        self.Down0_0 = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.conv0_0 = ConBlock(nf, nf)

        self.Down0_1 = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.conv0_1 = ConBlock(nf, nf)

        self.Down0_2 = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.conv0_2 = ConBlock(nf, nf)

        self.Up1 = nn.Sequential(
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.SKFF_1 = SKFF(in_channels=nf, height=2, reduction=8)
        self.Up2 = nn.Sequential(
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.SKFF_2 = SKFF(in_channels=nf, height=2, reduction=8)
        self.Up3 = nn.Sequential(
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
    def forward(self, input):
        x0_0 = self.conv0_0(self.Down0_0(input))
        x0_1 = self.conv0_1(self.Down0_1(x0_0))

        x0_2 = self.conv0_2(self.Down0_2(x0_1))
        up0_1 = self.Up1(x0_2)

        b,n,h,w = x0_1.shape
        up0_1 = up0_1[:,:,:h,:w]

        up0_2 = self.Up2(self.SKFF_1([up0_1, x0_1]))

        up0_3 = self.Up3(self.SKFF_1([up0_2, x0_0]))
        return up0_3+input


class OFAE(nn.Module):
    def __init__(self,  in_nc, out_nc, connection=False):
        super(OFAE, self).__init__()
        self.connection = connection
        if connection==True:
            self.decrease_dim = nn.Conv2d(in_nc, out_nc, 1, stride=1, padding=0)

        self.high_freq = nn.Sequential(
            nn.Conv2d(in_nc, out_nc, 3, stride=1, padding=3 // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.mid_freq = nn.Sequential(
            nn.Conv2d(in_nc, out_nc, 3, stride=2, padding=3 // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.low_freq = nn.Sequential(
            nn.Conv2d(in_nc, out_nc, 3, stride=4, padding=3 // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dcn_1 = nn.Sequential(
            nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.dcn_2 = nn.Sequential(
            nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.dcn_3 = nn.Sequential(
            nn.Conv2d(out_nc, out_nc, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.SKFF_1 = SKFF(in_channels=out_nc, height=2, reduction=8)
        self.SKFF_2 = SKFF(in_channels=out_nc, height=2, reduction=8)
        self.se = ChannelAttention(out_nc, reduction=32)
    def forward(self, x):
        f_l = self.low_freq(x)
        x_2 = self.mid_freq(x)
        x_1 = self.high_freq(x)
        f_m = x_2 - self.up(f_l)
        f_h = x_1 - self.up(x_2)

        f_l_enc = self.dcn_3(f_l)

        f_ml_enc = self.dcn_2(self.SKFF_1([self.up(f_l_enc), f_m]))
        f_mlh_enc = self.dcn_1(self.SKFF_2([self.up(f_ml_enc), f_h]))
        f_mlh_enc = self.se(f_mlh_enc)
        if self.connection==True:
            x = self.decrease_dim(x)
        return f_mlh_enc+x
