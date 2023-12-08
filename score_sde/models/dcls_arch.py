import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from utils import get_uperleft_denominator
from .module_util import *

class ResidualBlock_noBN(nn.Module):
    """Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    """

    def __init__(self, nf=64, res_scale=1.0):
        super(ResidualBlock_noBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        return identity + out.mul(self.res_scale)
class DPCAB(nn.Module):
    def __init__(self, nf1, nf2, ksize1=3, ksize2=3, reduction=4):
        super().__init__()

        self.body1 = nn.Sequential(
            nn.Conv2d(nf1, nf1, ksize1, 1, ksize1 // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf1, nf1, ksize1, 1, ksize1 // 2),
        )

        self.body2 = nn.Sequential(
            nn.Conv2d(nf2, nf2, ksize2, 1, ksize2 // 2),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf2, nf2, ksize2, 1, ksize2 // 2),
        )

        self.CA_body1 = nn.Sequential(
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf1+nf2, nf1, ksize1, 1, ksize1 // 2),
            CALayer(nf1, reduction))

        self.CA_body2 = CALayer(nf2, reduction)

    def forward(self, x):

        f1 = self.body1(x[0])
        f2 = self.body2(x[1])

        ca_f1 = self.CA_body1(torch.cat([f1, f2], dim=1))
        ca_f2 = self.CA_body2(f2)

        x[0] = x[0] + ca_f1
        x[1] = x[1] + ca_f2
        return x


class DPCAG(nn.Module):
    def __init__(self, nf1, nf2, ksize1, ksize2, nb):
        super().__init__()

        self.body = nn.Sequential(*[DPCAB(nf1, nf2, ksize1, ksize2) for _ in range(nb)])

    def forward(self, x):
        y = self.body(x)
        y[0] = x[0] + y[0]
        y[1] = x[1] + y[1]
        return y

class CLS(nn.Module):
    def __init__(self, nf, reduction=4):
        super().__init__()

        self.reduce_feature = nn.Conv2d(nf, nf//reduction, 1, 1, 0)

        self.grad_filter = nn.Sequential(
            nn.Conv2d(nf//reduction, nf//reduction, 3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf//reduction, nf//reduction, 3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf//reduction, nf//reduction, 3),
            nn.AdaptiveAvgPool2d((3, 3)),
            nn.Conv2d(nf//reduction, nf//reduction, 1),
        )

        self.expand_feature = nn.Conv2d(nf//reduction, nf, 1, 1, 0)

    def forward(self, x, kernel):
        cls_feats = self.reduce_feature(x)
        kernel_P = torch.exp(self.grad_filter(cls_feats))
        kernel_P = kernel_P - kernel_P.mean(dim=(2, 3), keepdim=True)
        clear_features = torch.zeros(cls_feats.size()).to(x.device)
        ks = kernel.shape[-1]
        dim = (ks, ks, ks, ks)
        feature_pad = F.pad(cls_feats, dim, "replicate")
        for i in range(feature_pad.shape[1]):
            feature_ch = feature_pad[:, i:i+1, :, :]
            clear_feature_ch = get_uperleft_denominator(feature_ch, kernel, kernel_P[:, i:i+1, :, :])
            clear_features[:, i:i+1, :, :] = clear_feature_ch[:, :, ks:-ks, ks:-ks]

        x = self.expand_feature(clear_features)

        return x

def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class Estimator(nn.Module):
    def __init__(
        self, in_nc=1, nf=64, para_len=10, num_blocks=3, kernel_size=4, filter_structures=[]
    ):
        super(Estimator, self).__init__()

        self.filter_structures = filter_structures
        self.ksize = kernel_size
        self.G_chan = 16
        self.in_nc = in_nc
        basic_block = functools.partial(ResidualBlock_noBN, nf=nf)

        self.head = nn.Sequential(
            nn.Conv2d(in_nc, nf, 7, 1, 3)
        )

        self.body = nn.Sequential(
            make_layer(basic_block, num_blocks)
        )

        self.tail = nn.Sequential(
            nn.Conv2d(nf, nf, 3),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nf, nf, 3),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(nf, para_len, 1),
            nn.Flatten(),
        )

        self.dec = nn.ModuleList()
        for i, f_size in enumerate(self.filter_structures):
            if i == 0:
                in_chan = in_nc
            elif i == len(self.filter_structures) - 1:
                in_chan = in_nc
            else:
                in_chan = self.G_chan
            self.dec.append(nn.Linear(para_len, self.G_chan * in_chan * f_size**2))

        self.apply(initialize_weights)

    def calc_curr_k(self, kernels, batch):
        """given a generator network, the function calculates the kernel it is imitating"""
        delta = torch.ones([1, batch*self.in_nc]).unsqueeze(-1).unsqueeze(-1).cuda()
        for ind, w in enumerate(kernels):
            curr_k = F.conv2d(delta, w, padding=self.ksize - 1, groups=batch) if ind == 0 else F.conv2d(curr_k, w, groups=batch)
        curr_k = curr_k.reshape(batch, self.in_nc, self.ksize, self.ksize).flip([2, 3])
        return curr_k

    def forward(self, LR):
        batch, channel = LR.shape[0:2]
        f1 = self.head(LR)
        f = self.body(f1) + f1

        latent_kernel = self.tail(f)

        kernels = [self.dec[0](latent_kernel).reshape(
                                                batch*self.G_chan,
                                                channel,
                                                self.filter_structures[0],
                                                self.filter_structures[0])]

        for i in range(1, len(self.filter_structures)-1):
            kernels.append(self.dec[i](latent_kernel).reshape(
                                                batch*self.G_chan,
                                                self.G_chan,
                                                self.filter_structures[i],
                                                self.filter_structures[i]))

        kernels.append(self.dec[-1](latent_kernel).reshape(
                                                batch*channel,
                                                self.G_chan,
                                                self.filter_structures[-1],
                                                self.filter_structures[-1]))

        K = self.calc_curr_k(kernels, batch).mean(dim=1, keepdim=True)

        # for anisox2
        # K = F.softmax(K.flatten(start_dim=1), dim=1)
        # K = K.view(batch, 1, self.ksize, self.ksize)

        K = K / torch.sum(K, dim=(2, 3), keepdim=True)

        return K

class SFTLayer(nn.Module):
    def __init__(self, in_nc=32, out_nc=64, nf=32):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(in_nc, nf, 1)
        self.SFT_scale_conv1 = nn.Conv2d(nf, out_nc, 1)
        self.SFT_shift_conv0 = nn.Conv2d(in_nc, nf, 1)
        self.SFT_shift_conv1 = nn.Conv2d(nf, out_nc, 1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(x[1]), 0.2, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(x[1]), 0.2, inplace=True))
        return x[0] * (scale + 1) + shift

def default_init_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)

class ResBlock(nn.Module):
    def __init__(self, in_feat, out_feat, stride=1):
        super(ResBlock, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_feat),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_feat),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_feat)
        )

    def forward(self, x):
        return nn.LeakyReLU(0.1, True)(self.backbone(x) + self.shortcut(x))

class ResBlock_with_SFT(nn.Module):
    def __init__(self, nf=64, in_nc=32, out_nc=64, time_emb_dim=0):
        super(ResBlock_with_SFT, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.out_nc = out_nc
        self.in_nc = in_nc
        self.sft1 = SFTLayer(in_nc=self.in_nc, out_nc=self.out_nc, nf=32)
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.sft2 = SFTLayer(in_nc=self.in_nc, out_nc=self.out_nc, nf=32)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1)

        # initialization
        default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        # x[0]: fea; x[1]: cond; x[2]: time
        fea = self.sft1((x[0], x[1]))
        fea = F.leaky_relu(self.conv1(fea), 0.2, inplace=True)
        fea = fea
        fea = self.sft2((fea, x[1]))
        fea = self.conv2(fea)
        return (x[0] + fea, x[1])

class Restorer(nn.Module):
    def __init__(
        self, in_nc=1, nf=64, nb=8, ng=1, scale=4, input_para=10, reduction=4, min=0.0, max=1.0, depths=[2, 2, 2, 8, 2, 2, 2]
    ):
        super(Restorer, self).__init__()
        self.min = min
        self.max = max
        self.para = input_para
        self.num_blocks = nb

        out_nc = in_nc
        nf2 = nf // reduction

        self.conv_first = nn.Conv2d(in_nc, nf, 3, stride=1, padding=1)
        basic_block = functools.partial(ResidualBlock_noBN, nf=nf)
        self.feature_block = make_layer(basic_block, 3)

        self.head1 = nn.Conv2d(nf, nf2, 3, 1, 1)
        self.head2 = CLS(nf, reduction=reduction)

        # body = [DPCAG(nf, nf2, 3, 3, nb) for _ in range(ng)]
        # self.body = nn.Sequential(*body)

        # self.fusion = nn.Conv2d(nf+nf2, nf, 3, 1, 1)
        # self.fusion = CCALayer(nf, nf, reduction)

#         if scale == 4:  # x4
#             self.upscale = nn.Sequential(
#                 nn.Conv2d(nf,nf * scale,3,1,1,bias=True),
#                 nn.PixelShuffle(scale // 2),
#                 nn.Conv2d(nf,nf * scale,3,1,1,bias=True),
#                 nn.PixelShuffle(scale // 2),
#                 nn.Conv2d(nf, out_nc, 3, 1, 1),
#             )
#         elif scale == 1:
#             self.upscale = nn.Conv2d(nf, out_nc, 3, 1, 1)

#         else:  # x2, x3
#             self.upscale = nn.Sequential(
#                 nn.Conv2d(nf,nf * scale ** 2,3,1,1,bias=True),
#                 nn.PixelShuffle(scale),
#                 nn.Conv2d(nf, out_nc, 3, 1, 1),
#             )

#         # self.relu = nn.LeakyReLU(0.1, inplace=True)
#         self.ResBlock = ResBlock(nf2, nf, stride=1)

#         self.CondNet0 = nn.Sequential(nn.Conv2d(nf, nf, 1), nn.LeakyReLU(0.2, True),
#                                       nn.Conv2d(nf, nf, 1), nn.LeakyReLU(0.2, True),
#                                       nn.Conv2d(nf, nf, 1), nn.LeakyReLU(0.2, True),
#                                       nn.Conv2d(nf, nf, 1))

#         self.CondNet1 = nn.Sequential(nn.Conv2d(nf, nf, 3, 2, 1), nn.LeakyReLU(0.2, True),
#                                       nn.Conv2d(nf, nf, 1), nn.LeakyReLU(0.2, True),
#                                       nn.Conv2d(nf, nf, 1), nn.LeakyReLU(0.2, True),
#                                       nn.Conv2d(nf, nf * 2, 1))

#         self.CondNet2 = nn.Sequential(nn.Conv2d(nf, nf, 3, 2, 1), nn.LeakyReLU(0.2, True),
#                                       nn.Conv2d(nf, nf, 3, 2, 1), nn.LeakyReLU(0.2, True),
#                                       nn.Conv2d(nf, nf, 1), nn.LeakyReLU(0.2, True),
#                                       nn.Conv2d(nf, nf * 4, 1))

#         self.CondNet3 = nn.Sequential(nn.Conv2d(nf, nf, 3, 2, 1), nn.LeakyReLU(0.2, True),
#                                       nn.Conv2d(nf, nf, 3, 2, 1), nn.LeakyReLU(0.2, True),
#                                       nn.Conv2d(nf, nf, 3, 2, 1), nn.LeakyReLU(0.2, True),
#                                       nn.Conv2d(nf, nf * 8, 1))

#         ## Base
#         self.conv_first2 = nn.Sequential(nn.Conv2d(nf+nf2, nf, 3, 1, 1), nn.LeakyReLU(0.2, True))
#         basic_block = functools.partial(ResBlock_with_SFT, nf=nf, in_nc=nf, out_nc=nf)
#         basic_block2 = functools.partial(ResBlock_with_SFT, nf=nf * 2, in_nc=nf * 2, out_nc=nf * 2)
#         basic_block4 = functools.partial(ResBlock_with_SFT, nf=nf * 4, in_nc=nf * 4, out_nc=nf * 4)
#         basic_block8 = functools.partial(ResBlock_with_SFT, nf=nf * 8, in_nc=nf * 8, out_nc=nf * 8)

#         self.enconv_layer0 = make_layer(basic_block, depths[0])
#         self.down_conv0 = nn.Conv2d(nf, nf * 2, 3, 2, 1)

#         self.enconv_layer1 = make_layer(basic_block2, depths[1])
#         self.down_conv1 = nn.Conv2d(nf * 2, nf * 4, 3, 2, 1)

#         self.enconv_layer2 = make_layer(basic_block4, depths[2])
#         self.down_conv2 = nn.Conv2d(nf * 4, nf * 8, 3, 2, 1)

#         self.Bottom_conv = make_layer(basic_block8, depths[3])

#         self.up_conv2 = nn.Sequential(nn.Conv2d(nf * 8, nf * 4 * 4, 3, 1, 1), nn.PixelShuffle(2))
#         self.deconv_layer2 = make_layer(basic_block4, depths[4])

#         self.up_conv1 = nn.Sequential(nn.Conv2d(nf * 4, nf * 2 * 4, 3, 1, 1), nn.PixelShuffle(2))
#         self.deconv_layer1 = make_layer(basic_block2, depths[5])

#         self.up_conv0 = nn.Sequential(nn.Conv2d(nf * 2, nf * 4, 3, 1, 1), nn.PixelShuffle(2))
#         self.deconv_layer0 = make_layer(basic_block, depths[6])

#         self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

    def forward(self, input, kernel):
        # B, C, H, W = input.size()  # I_LR batch

        f = self.conv_first(input)
        feature = self.feature_block(f)
        f1 = self.head1(feature)
        f2 = self.head2(feature, kernel) #下采样特征

#         cond = self.ResBlock(f1)
#         cond0 = self.CondNet0(cond)
#         cond1 = self.CondNet1(cond)
#         cond2 = self.CondNet2(cond)
#         cond3 = self.CondNet3(cond)

#         x1 = torch.cat((f2, f1), 1)
#         fea0 = self.conv_first2(x1)

#         fea0, _ = self.enconv_layer0((fea0, cond0))
#         down0 = self.down_conv0(fea0)

#         fea1, _ = self.enconv_layer1((down0, cond1))
#         down1 = self.down_conv1(fea1)

#         fea2, _ = self.enconv_layer2((down1, cond2))
#         down2 = self.down_conv2(fea2)
#         # print(down2.shape)
#         # print(kernel3.shape)
#         feaB, _ = self.Bottom_conv((down2, cond3))
#         feaB = feaB + down2

#         up2 = self.up_conv2(feaB) + fea2
#         defea2, _ = self.deconv_layer2((up2, cond2))

#         up1 = self.up_conv1(defea2) + fea1
#         defea1, _ = self.deconv_layer1((up1, cond1))

#         up0 = self.up_conv0(defea1) + fea0
#         defea0, _ = self.deconv_layer0((up0, cond0))

#         # out = self.conv_last(defea0)
#         # inputs = [f2, f1]
#         # f2, f1 = self.body(inputs)
#         # f = self.fusion(torch.cat([f1, f2], dim=1)) + f
#         out = self.upscale(defea0)
        return f1, f2


class DCLS(nn.Module):
    def __init__(
        self,
        nf=64,
        nb=16,
        ng=5,
        in_nc=3,
        reduction=4,
        upscale=4,
        input_para=256,
        kernel_size=31,
        pca_matrix_path=None,
    ):
        super(DCLS, self).__init__()

        self.ksize = kernel_size
        self.scale = upscale

        if kernel_size == 21:
            filter_structures = [11, 7, 5, 1] # for iso kernels all
        elif kernel_size == 11:
            filter_structures = [7, 3, 3, 1] # for aniso kernels x2
        elif kernel_size == 31:
            filter_structures = [11, 9, 7, 5, 3] # for aniso kernels x4
        else:
            print("Please check your kernel size, or reset a group filters for DDLK")

        self.Restorer = Restorer(
            nf=nf, in_nc=in_nc, nb=nb, ng=ng, scale=self.scale, input_para=input_para, reduction=reduction
        )
        self.Estimator = Estimator(
            kernel_size=kernel_size, para_len=input_para, in_nc=in_nc, nf=nf, filter_structures=filter_structures
        )

    def forward(self, lr):

        kernel = self.Estimator(lr)
        f1, f2 = self.Restorer(lr, kernel.detach())

        return f1, f2
