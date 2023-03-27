import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d
import functools
from base_model import OFAE, SKU_Net
import numpy as np


def generate_it(x, t=0, nf=3, f=7):
    index = np.array([t - nf // 2 + i for i in range(nf)])
    index = np.clip(index, 0, f-1).tolist()
    it = x[:, index, :, :]
    return it

class STFF(nn.Module):
    def __init__(self, in_nc, out_nc, nf, base_ks=3, deform_ks=3):
        """
        Args:
            in_nc: num of input channels.
            out_nc: num of output channels.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            deform_ks: size of the deformable kernel.
        """
        super(STFF, self).__init__()

        self.in_nc = in_nc
        self.deform_ks = deform_ks
        self.size_dk = deform_ks ** 2

        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=base_ks // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.unet = SKU_Net(nf=nf)

        self.out_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.offset_mask = nn.Conv2d(
            nf, in_nc * 3 * self.size_dk, base_ks, padding=base_ks // 2
        )

        self.deform_conv = DeformConv2d(
            in_nc, out_nc, deform_ks, padding=deform_ks // 2
        )

    def forward(self, inputs):
        out = self.in_conv(inputs)
        out = self.unet(out)
        off_msk = self.offset_mask(self.out_conv(out))
        off = off_msk[:, :self.in_nc * 2 * self.size_dk, ...]
        msk = torch.sigmoid(
            off_msk[:, self.in_nc * 2 * self.size_dk:, ...]
        )
        fused_feat = F.relu(
            self.deform_conv(inputs, off, msk),
            inplace=True
        )
        return fused_feat



class PlainCNN(nn.Module):
    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)
    def __init__(self, in_nc=64, nf=64, nb=5, out_nc=3, base_ks=3):
        """
        Args:
            in_nc: num of input channels from STDF.
            nf: num of channels (filters) of each conv layer.
            nb: num of conv layers.
            out_nc: num of output channel. 3 for RGB, 1 for Y.
        """
        super(PlainCNN, self).__init__()
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_nc, nf, base_ks, padding=1),
            nn.ReLU(inplace=True)
        )
        self.reconstruct = self.make_layer(functools.partial(OFAE, nf, nf), nb)
        self.out_conv = nn.Conv2d(nf, out_nc, base_ks, padding=1)

    def forward(self, inputs):
        inputs = self.in_conv(inputs)
        inputs = self.reconstruct(inputs)
        inputs = self.out_conv(inputs)
        return inputs


class OVQE(nn.Module):
    def __init__(self, in_nc=7, nf=32, out_nc=64, nb=5, cpu_cache_length=20):
        super(OVQE, self).__init__()
        self.out_nc = out_nc
        self.cpu_cache_length = cpu_cache_length
        self.stff = STFF(in_nc=in_nc, out_nc=out_nc, nf=nf, deform_ks=3)
        self.first_backward = STFF(
            in_nc=2*out_nc,
            out_nc=out_nc,
            nf=nf,
            deform_ks=1
        )
        self.first_forward = STFF(
            in_nc=3*out_nc,
            out_nc=out_nc,
            nf=nf,
            deform_ks=1
        )
        self.second_backward = STFF(
            in_nc=3*out_nc,
            out_nc=out_nc,
            nf=nf,
            deform_ks=1
        )
        self.second_forward = STFF(
            in_nc=3*out_nc,
            out_nc=out_nc,
            nf=nf,
            deform_ks=1
        )
        self.ofae = OFAE(2*self.out_nc,self.out_nc, connection=True)
        self.qenet = PlainCNN(in_nc=4*self.out_nc,nf=self.out_nc,nb=nb,out_nc=1)


    def forward(self, inputs):
        n, t, h, w = inputs.size()
        if t > self.cpu_cache_length:
            self.cpu_cache = True
        else:
            self.cpu_cache = False
#####################################First Backward Propagation############################################
        First_Backward_List = []
        feat = inputs.new_zeros(n, self.out_nc, h, w)
        for i in range(t - 1, -1, -1):
            out = generate_it(inputs, i, 7, t)
            out = self.stff(out)
            feat = torch.cat([out,feat], dim=1)
            feat = self.first_backward(feat)
            feat = self.ofae(torch.cat([out,feat], 1)) + out
            if self.cpu_cache:
                First_Backward_List.append(feat.cpu())
                torch.cuda.empty_cache()
            else:
                First_Backward_List.append(feat)
        First_Backward_List = First_Backward_List[::-1]
#####################################First Forward Propagation##############################################
        First_Forward_List = []
        feat = inputs.new_zeros(n, self.out_nc, h, w)
        for i in range(0, t):
            future = First_Backward_List[i] if i == t - 1 else First_Backward_List[i + 1]
            present = First_Backward_List[i]
            if self.cpu_cache:
                present = present.cuda()
                future = future.cuda()

            feat = torch.cat([feat,present,future], dim=1)
            feat = self.first_forward(feat)
            feat = self.ofae(torch.cat([present, feat], 1)) + present
            if self.cpu_cache:
                First_Forward_List.append(feat.cpu())
                torch.cuda.empty_cache()
            else:
                First_Forward_List.append(feat)
#####################################Second Backward Propagation##########################################
        Second_Backward_List = []
        feat = inputs.new_zeros(n, self.out_nc, h, w)
        for i in range(t - 1, -1, -1):
            future = First_Forward_List[i] if i == 0 else First_Forward_List[i - 1]
            present = First_Forward_List[i]
            if self.cpu_cache:
                present = present.cuda()
                future = future.cuda()

            feat = torch.cat([feat,present,future], dim=1)
            feat = self.second_backward(feat)
            feat = self.ofae(torch.cat([present, feat], 1))  + present
            if self.cpu_cache:
                Second_Backward_List.append(feat.cpu())
                torch.cuda.empty_cache()
            else:
                Second_Backward_List.append(feat)
        Second_Backward_List = Second_Backward_List[::-1]
#####################################Second Forward Propagation############################################
        Enhanced = []
        feat = inputs.new_zeros(n, self.out_nc, h, w)
        for i in range(0, t):
            future = Second_Backward_List[i] if i == t - 1 else Second_Backward_List[i + 1]
            present = Second_Backward_List[i]
            if self.cpu_cache:
                present = present.cuda()
                future = future.cuda()

            feat = torch.cat([feat,present,future], dim=1)
            feat = self.second_forward(feat)
            feat = self.ofae(torch.cat([present, feat], 1))+ present
            if self.cpu_cache:
                out = self.qenet(torch.cat([First_Backward_List[i].cuda(), First_Forward_List[i].cuda(), Second_Backward_List[i].cuda(), feat],dim=1)) + inputs[:, i:i + 1, :, :]
                Enhanced.append(out.cpu())
                torch.cuda.empty_cache()
            else:
                out = self.qenet(torch.cat([First_Backward_List[i], First_Forward_List[i], Second_Backward_List[i], feat],dim=1)) + inputs[:, i:i + 1, :, :]
                Enhanced.append(out)

        return torch.stack(Enhanced, dim=1)

if __name__ == "__main__":
    torch.cuda.set_device(1)
    net = OVQE().cuda()
    from thop import profile
    with torch.no_grad():

        input = torch.randn(1, 15, 32, 32).cuda()
        flops, params = profile(net, inputs=(input, ))
        total = sum([param.nelement() for param in net.parameters()])
        print('   Number of params: %.2fM' % (total / 1e6))
        print('   Number of FLOPs: %.2fGFLOPs' % (flops / 1e9))
