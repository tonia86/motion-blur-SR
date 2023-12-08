from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from . import dense_layer
from . import layers
import torch

dense = dense_layer.dense
get_sinusoidal_positional_embedding = layers.get_timestep_embedding

class TimestepEmbedding(nn.Module):#位置编码，全连接层-激活层-全连接层
    def __init__(self, embedding_dim, hidden_dim, output_dim, act=nn.LeakyReLU(0.2)):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.main = nn.Sequential(
            dense(embedding_dim, hidden_dim),
            act,
            dense(hidden_dim, output_dim),
        )

    def forward(self, temp):
        temb = get_sinusoidal_positional_embedding(temp, self.embedding_dim)
        temb = self.main(temb)
        return temb

class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)
    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.
    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64,t_emb_dim=128, skip_connection=True, act=nn.LeakyReLU(0.2)):
        super(UNetDiscriminatorSN, self).__init__()
        self.act = act
        self.t_embed = TimestepEmbedding(
            embedding_dim=t_emb_dim,
            hidden_dim=t_emb_dim,
            output_dim=t_emb_dim,
            act=act,
        )
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.dense_t0 = dense(t_emb_dim, num_feat)
        self.conv0 = nn.Conv2d(num_in_ch*2, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.dense_t1 = dense(t_emb_dim, num_feat * 2)
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.dense_t2= dense(t_emb_dim, num_feat * 4)
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        self.dense_t3 = dense(t_emb_dim, num_feat * 8)
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.dense_t4 = dense(t_emb_dim, num_feat * 4)
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.dense_t5 = dense(t_emb_dim, num_feat * 2)
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        self.dense_t6 = dense(t_emb_dim, num_feat)
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x, t_emb,  x_t):
        # downsample
        t_emb = self.act(self.t_embed(t_emb))
        input_x = torch.cat((x, x_t), dim=1)
        x0 = F.leaky_relu(self.conv0(input_x), negative_slope=0.2, inplace=False)
        x0 += self.dense_t0(t_emb)[..., None, None]
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=False)
        x1 += self.dense_t1(t_emb)[..., None, None]
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=False)
        x2 += self.dense_t2(t_emb)[..., None, None]
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=False)
        x3 += self.dense_t3(t_emb)[..., None, None]

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=False)
        x4 += self.dense_t4(t_emb)[..., None, None]
        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=False)
        x5 += self.dense_t5(t_emb)[..., None, None]

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=False)
        x6 += self.dense_t6(t_emb)[..., None, None]

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=False)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=False)
        out = self.conv9(out)

        return out