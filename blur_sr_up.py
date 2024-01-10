# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


import argparse
import torch
import numpy as np

# import os

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

import logging
import torchvision.transforms as transforms

from torchvision.utils import make_grid
import math
from torch.multiprocessing import Process
import torch.distributed as dist
import shutil

###jia###
import options as option
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler
import utils as util
from data.util import bgr2ycbcr
import lpips
#import RRDBNet_arch as arch
from data.dataset_hf5 import DataSet, DataValSet

from torch.utils.data import DataLoader
import os
import random
from os.path import join

def is_hdf5_file(filename):
    return any(filename.endswith(extension) for extension in [".h5"])

def convert_otf2psf(otf, size):
    ker = torch.zeros(size).cuda()
    psf = torch.irfft(otf, 3, onesided=False)

    # circularly shift
    ksize = size[-1]
    centre = ksize//2 + 1

    ker[:, :, (centre-1):, (centre-1):] = psf[:, :, :centre, :centre]#.mean(dim=1, keepdim=True)
    ker[:, :, (centre-1):, :(centre-1)] = psf[:, :, :centre, -(centre-1):]#.mean(dim=1, keepdim=True)
    ker[:, :, :(centre-1), (centre-1):] = psf[:, :, -(centre-1):, :centre]#.mean(dim=1, keepdim=True)
    ker[:, :, :(centre-1), :(centre-1)] = psf[:, :, -(centre-1):, -(centre-1):]#.mean(dim=1, keepdim=True)

    return ker

def normkernel_to_downkernel(rescaled_blur_hr, rescaled_hr, ksize, eps=1e-10):
    blur_img = torch.rfft(rescaled_blur_hr, 3, onesided=False)
    img = torch.rfft(rescaled_hr, 3, onesided=False)

    denominator = img[:, :, :, :, 0] * img[:, :, :, :, 0] \
                  + img[:, :, :, :, 1] * img[:, :, :, :, 1] + eps

    # denominator[denominator==0] = eps

    inv_denominator = torch.zeros_like(img)
    inv_denominator[:, :, :, :, 0] = img[:, :, :, :, 0] / denominator
    inv_denominator[:, :, :, :, 1] = -img[:, :, :, :, 1] / denominator

    kernel = torch.zeros_like(blur_img).cuda()
    kernel[:, :, :, :, 0] = inv_denominator[:, :, :, :, 0] * blur_img[:, :, :, :, 0] \
                            - inv_denominator[:, :, :, :, 1] * blur_img[:, :, :, :, 1]
    kernel[:, :, :, :, 1] = inv_denominator[:, :, :, :, 0] * blur_img[:, :, :, :, 1] \
                            + inv_denominator[:, :, :, :, 1] * blur_img[:, :, :, :, 0]

    ker = convert_otf2psf(kernel, ksize)

    return ker

def zeroize_negligible_val(k, n=40):
    """Zeroize values that are negligible w.r.t to values in k"""
    # Sort K's values in order to find the n-th largest
    pc = k.shape[-1]//2 + 1
    k_sorted, indices = torch.sort(k.flatten(start_dim=1))
    # Define the minimum value as the 0.75 * the n-th largest value
    k_n_min = 0.75 * k_sorted[:, -n - 1]
    # Clip values lower than the minimum value
    filtered_k = torch.clamp(k - k_n_min.view(-1, 1, 1, 1), min=0, max=1.0)
    filtered_k[:, :, pc, pc] += 1e-20
    # Normalize to sum to 1
    norm_k = filtered_k / torch.sum(filtered_k, dim=(2, 3), keepdim=True)
    return norm_k

def postprocess(*images, rgb_range):
    def _postprocess(img):
        pixel_range = 255 / rgb_range
        return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

    return [_postprocess(img) for img in images]

class CorrectionLoss(nn.Module):
    def __init__(self, scale=4.0, eps=1e-6):
        super(CorrectionLoss, self).__init__()
        self.scale = scale
        self.eps = eps
        self.cri_pix = nn.L1Loss()

    def forward(self, k_pred, lr_blured, lr):

        ks = []
        mask = torch.ones_like(k_pred).cuda()
        for c in range(lr_blured.shape[1]):
            k_correct = normkernel_to_downkernel(lr_blured[:, c:c+1, ...], lr[:, c:c+1, ...], k_pred.size(), self.eps)
            ks.append(k_correct.clone())
            mask *= k_correct
        ks = torch.cat(ks, dim=1)
        k_correct = torch.mean(ks, dim=1, keepdim=True) * (mask>0)
        k_correct = zeroize_negligible_val(k_correct, n=40)

        return self.cri_pix(k_pred, k_correct), k_correct
    
# gaussian diffusion trainer class
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


def _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, warmup_frac):
    betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    warmup_time = int(num_diffusion_timesteps * warmup_frac)
    betas[:warmup_time] = np.linspace(beta_start, beta_end, warmup_time, dtype=np.float64)
    return betas


def get_beta_schedule(num_diffusion_timesteps, beta_schedule='linear', beta_start=0.0001, beta_end=0.02):
    if beta_schedule == 'quad':
        betas = np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2
    elif beta_schedule == 'linear':
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'warmup10':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.1)
    elif beta_schedule == 'warmup50':
        betas = _warmup_beta(beta_start, beta_end, num_diffusion_timesteps, 0.5)
    elif beta_schedule == 'const':
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


def create_logger(fp):
    # 打印日志的时间、日志级别名称、日志信息
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
    console_logger = logging.getLogger('ConsoleLoggoer')
    file_logger = logging.getLogger('FileLogger')

    # 向文件输出日志信息
    file_handler = logging.FileHandler(fp, mode='a', encoding='utf-8')
    file_logger.addHandler(file_handler)
    return console_logger, file_logger


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))


def broadcast_params(params):
    for param in params:
        dist.broadcast(param.data, src=0)


# %% Diffusion coefficients
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var  # 计算的就是sigma的平方


def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)  # 在论文中没有使用


def extract(input, t, shape):  # 根据t取对应位置的input
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)  # [3] + [1] * 2的结果为[3, 1, 1]
    out = out.reshape(*reshape)

    return out


def get_time_schedule(args, device):
    n_timestep = args.num_timesteps
    eps_small = 1e-3
    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small
    return t.to(device)


def get_sigma_schedule(args, device):
    n_timestep = args.num_timesteps  # num_timesteps默认值为4
    beta_min = args.beta_min  # 最小值为0.1
    beta_max = args.beta_max  # 最大值为20
    eps_small = 1e-3

    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep  # 对时间进行归一化
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small  # 这里无穷小量的意义在哪里

    if args.use_geometric:  # 默认为false
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)  # 计算得到方差的平方
    alpha_bars = 1.0 - var  # alpha_t一拔
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]  # beta_t

    first = torch.tensor(1e-8)  # 表示的是10的-8次方
    betas = torch.cat((first[None], betas)).to(device)  # 在求得的beta的最前面加入一个无穷小值
    betas = betas.type(torch.float32)
    sigmas = betas ** 0.5  # 标准差
    a_s = torch.sqrt(1 - betas)
    return sigmas, a_s, betas


class Diffusion_Coefficients():  # 扩散系数
    def __init__(self, args, device):
        self.sigmas, self.a_s, _ = get_sigma_schedule(args, device=device)
        self.a_s_cum = np.cumprod(self.a_s.cpu())  # 根号下αt的连乘
        self.sigmas_cum = np.sqrt(1 - self.a_s_cum ** 2)  # 根号下1-(αt的连乘)
        self.a_s_prev = self.a_s.clone()
        self.a_s_prev[-1] = 1

        self.a_s_cum = self.a_s_cum.to(device)  # 将数据传到gpu中
        self.sigmas_cum = self.sigmas_cum.to(device)
        self.a_s_prev = self.a_s_prev.to(device)


def q_sample(coeff, x_start, t, *, noise=None):  # 由x0，计算xt
    """
    Diffuse the data (t == 0 means diffused for t step)
    """
    if noise is None:
        noise = torch.randn_like(x_start)

    x_t = extract(coeff.a_s_cum, t, x_start.shape) * x_start + \
          extract(coeff.sigmas_cum, t, x_start.shape) * noise  # 即破坏图像，得到xt的计算公式

    return x_t


def t(img):
    def to_4d(img):
        assert len(img.shape) == 3
        img_new = np.expand_dims(img, axis=0)
        assert len(img_new.shape) == 4
        return img_new

    def to_CHW(img):
        return np.transpose(img, [2, 0, 1])

    def to_tensor(img):
        return torch.Tensor(img)

    return to_tensor(to_4d(to_CHW(img))) / 127.5 - 1


def Lpips(imgA, imgB):
    model = lpips.LPIPS(net='alex')
    device = next(model.parameters()).device
    tA = t(imgA).to(device)
    tB = t(imgB).to(device)
    dist01 = model.forward(tA, tB).item()
    return dist01


def q_sample_pairs(coeff, x_start, t):  # 输入真实图像，计算破坏后的图像，xt，xt+1
    """
    Generate a pair of disturbed images for training
    :param x_start: x_0
    :param t: time step t
    :return: x_t, x_{t+1}
    """
    noise = torch.randn_like(x_start)
    x_t = q_sample(coeff, x_start, t)  # 由x0，计算xt
    x_t_plus_one = extract(coeff.a_s, t + 1, x_start.shape) * x_t + \
                   extract(coeff.sigmas, t + 1, x_start.shape) * noise  # 由xt，计算xt+1

    return x_t, x_t_plus_one, noise


# %% posterior sampling sqrt_recip_alphas_cumprod sqrt_recipm1_alphas_cumprod
class Posterior_Coefficients():
    def __init__(self, args, device):
        _, _, self.betas = get_sigma_schedule(args, device=device)

        # we don't need the zeros
        self.betas = self.betas.type(torch.float32)[1:]

        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.cat(
            (torch.tensor([1.], dtype=torch.float32, device=device), self.alphas_cumprod[:-1]), 0
        )
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.sqrt_alphas_cumprod_one = torch.sqrt(1-self.alphas_cumprod)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)

        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))

        self.posterior_mean_coef2 = (
                (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))

        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))


def bgr2ycbcr(img, only_y=True):
    '''bgr version of rgb2ycbcr
    only_y: only return Y channel
    Input:
        uint8, [0, 255]
        float, [0, 1]
    '''
    in_img_type = img.dtype
    img.astype(np.float32)
    if in_img_type != np.uint8:
        img *= 255.
    # convert
    if only_y:
        rlt = np.dot(img, [24.966, 128.553, 65.481]) / 255.0 + 16.0
    else:
        rlt = np.matmul(img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                              [65.481, -37.797, 112.0]]) / 255.0 + [16, 128, 128]
    if in_img_type == np.uint8:
        rlt = rlt.round()
    else:
        rlt /= 255.
    return rlt.astype(in_img_type)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    """
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    """
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            "Only support 4D, 3D and 2D tensor. But received with dimension: {:d}".format(
                n_dim
            )
        )
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def sample_posterior(coefficients, x_0, x_t, t):
    def q_posterior(x_0, x_t, t):
        mean = (
                extract(coefficients.posterior_mean_coef1, t, x_t.shape) * x_0
                + extract(coefficients.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(coefficients.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(coefficients.posterior_log_variance_clipped, t, x_t.shape)
        return mean, var, log_var_clipped

    def p_sample(x_0, x_t, t):
        mean, _, log_var = q_posterior(x_0, x_t, t)

        noise = torch.randn_like(x_t)

        nonzero_mask = (1 - (t == 0).type(torch.float32))

        return mean + nonzero_mask[:, None, None, None] * torch.exp(0.5 * log_var) * noise

    sample_x_pos = p_sample(x_0, x_t, t)

    return sample_x_pos


def sample_from_model(coefficients, generator, n_time, x_init, lr_data_down, T, opt):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)

            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0, _ = generator(x, t_time, lr_data_down)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()

    return x


# %%
def train(rank, gpu, args):
    from score_sde.models.discriminator import Discriminator_small, Discriminator_large
    from score_sde.models.ncsnpp_blur_sr_gfn import Unet
    from EMA import EMA
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))
    console_logger, file_logger = create_logger(os.path.join('./', 'train_motionblur.log'))

    #### distributed training settings
    opt["dist"] = True

    torch.backends.cudnn.benchmark = True

    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    batch_size = args.batch_size

    #####training dataset#####
    train_dir = '/tn/Data/GOPRO_train256_4x_HDF5/' #在此修改数据集的路径
    train_sets = [x for x in sorted(os.listdir(train_dir)) if is_hdf5_file(x)] 
    
    #####test dataset#####
    root_val_dir = '/tn/Data/Validation_4x/'# #----------Validation path
    testloader = DataLoader(DataValSet(root_val_dir), batch_size=1, shuffle=False, pin_memory=False)
 
    ####
    hidden_size = args.hidden_size
    dim_mults = args.ch_mult
    netG = Unet(opt, 64, out_dim=3, cond_dim=32, dim_mults=dim_mults).to(device)

    netD = Discriminator_large(nc=2 * args.num_channels, ngf=args.ngf,
                               t_emb_dim=args.t_emb_dim,
                               act=nn.LeakyReLU(0.2)).to(device)

    broadcast_params(netG.parameters())
    broadcast_params(netD.parameters())

    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    optimizerG = optim.Adam(netG.parameters(), lr=args.lr_g, betas=(args.beta1, args.beta2))

    if args.use_ema:
        optimizerG = EMA(optimizerG, ema_decay=args.ema_decay)

    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, args.num_epoch, eta_min=1e-5)
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, args.num_epoch, eta_min=1e-5)

    # ddp
    netG = nn.parallel.DistributedDataParallel(netG, device_ids=[gpu], find_unused_parameters=True)
    netD = nn.parallel.DistributedDataParallel(netD, device_ids=[gpu], find_unused_parameters=True)

    exp = args.exp  # 实验的名称 experiment_cifar_default
    parent_dir = "./saved_info/dd_gan/{}".format(args.dataset)

    exp_path = os.path.join(parent_dir, exp)
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
            copy_source(__file__, exp_path)
            shutil.copytree('score_sde/models', os.path.join(exp_path, 'score_sde/models'))
            # 递归地将以 src 为根起点的整个目录树拷贝到名为 dst 的目录并返回目标目录。 所需的包含 dst 的中间目录在默认情况下也将被创建。

    coeff = Diffusion_Coefficients(args, device)  # 扩散系数
    pos_coeff = Posterior_Coefficients(args, device)  # 后验系数
    T = get_time_schedule(args, device)

    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        epoch = init_epoch
        netG.load_state_dict(checkpoint['netG_dict'])
        # load G

        optimizerG.load_state_dict(checkpoint['optimizerG'])
        schedulerG.load_state_dict(checkpoint['schedulerG'])
        # load D
        netD.load_state_dict(checkpoint['netD_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        schedulerD.load_state_dict(checkpoint['schedulerD'])
        global_step = checkpoint['global_step']
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0

    print('num_epoch:', args.num_epoch)
    to_range_1_1 = lambda x: 2.*x-1.
    to_range_0_1 = lambda x: (x + 1.) / 2.
    for epoch in range(init_epoch, args.num_epoch + 1):
        # train_sampler.set_epoch(epoch)#在分布式模式下，需要在每个 epoch 开始时调用 set_epoch() 方法，然后再创建 DataLoader 迭代器，
        # 以使 shuffle 操作能够在多个 epoch 中正常工作。 否则，dataloader迭代器产生的数据将始终使用相同的顺序。
        random.shuffle(train_sets) #对数据集中的图像进行打乱
        for j in range(len(train_sets)):
#             print("Step {}:Training folder is {}-------------------------------".format(i, train_sets[j]))
            train_set = DataSet(join(train_dir, train_sets[j]))
            trainloader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=1)
            for iteration, batch in enumerate(trainloader, 1):
            #input, targetdeblur, targetsr
                lr_data_down = batch[0]
                lr_data_down = to_range_1_1(lr_data_down)
                lr_data_deblur = batch[1]
                lr_data_deblur = to_range_1_1(lr_data_deblur)
                real_data = batch[2]
                real_data = to_range_1_1(real_data)
                lr_data_down = lr_data_down.to(device)
                lr_data_deblur = lr_data_deblur.to(device)
                real_data = real_data.to(device)
                
                for p in netD.parameters():
                    p.requires_grad = True

                netD.zero_grad()

                # sample t
                t = torch.randint(0, args.num_timesteps, (real_data.size(0),),
                                  device=device)  # 生成一个（x,）大小的随机数范围为0到num_timesteps

                x_t, x_tp1, noise = q_sample_pairs(coeff, real_data, t)  # 计算破坏后的图像xt，xt+1
                
                x_t.requires_grad = True

                # train with real
                D_real = netD(x_t, t, x_tp1.detach()).view(-1)  # 将真实数据输入到判别器中
                torchvision.utils.save_image(x_t,
                                             os.path.join(exp_path, 'x_t_{}.png'.format(iteration)),
                                             normalize=True)
                errD_real = F.softplus(-D_real)  # 平滑版的relu函数
                errD_real = errD_real.mean()

                errD_real.backward(retain_graph=True)

                if args.lazy_reg is None:
                    grad_real = torch.autograd.grad(
                        outputs=D_real.sum(), inputs=x_t, create_graph=True
                    )[0]
                    grad_penalty = (
                            grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                    ).mean()

                    grad_penalty = args.r1_gamma / 2 * grad_penalty
                    grad_penalty.backward()
                else:
                    if global_step % args.lazy_reg == 0:
                        grad_real = torch.autograd.grad(
                            outputs=D_real.sum(), inputs=x_t, create_graph=True
                        )[0]
                        grad_penalty = (
                                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                        ).mean()

                        grad_penalty = args.r1_gamma / 2 * grad_penalty
                        grad_penalty.backward()

                # train with fake (self, x, time, cond, img_lr_up)
                with torch.no_grad():
                    epsilon_t, _ = netG(x_tp1.detach(), t, lr_data_down)

                x_pos_sample = sample_posterior(pos_coeff, epsilon_t, x_tp1, t)

                output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)

                errD_fake = F.softplus(output)
                errD_fake = errD_fake.mean()
                errD_fake.backward()

                errD = errD_real + errD_fake
                # Update D
                optimizerD.step()

                # update G
                for p in netD.parameters():
                    p.requires_grad = False
                netG.zero_grad()

                epsilon_t, kernel_sr = netG(x_tp1.detach(), t, lr_data_down)
                x_pos_sample = sample_posterior(pos_coeff, epsilon_t, x_tp1, t)
                torchvision.utils.save_image(x_pos_sample,
                                             os.path.join(exp_path, 'x0_{}.png'.format(iteration)),
                                             normalize=True)
                torchvision.utils.save_image(kernel_sr,
                                             os.path.join(exp_path, 'kernel_sr_{}.png'.format(iteration)),
                                             normalize=True)
                output = netD(x_pos_sample, t, x_tp1.detach()).view(-1)

                errG = F.softplus(-output)
                errG = errG.mean()

                # reconstructior loss
                if args.rec_loss:
                    loss_k = F.l1_loss(kernel_sr, lr_data_deblur)
                    Rec_loss = F.l1_loss(epsilon_t, real_data) 
                    errG = 0.001*errG + Rec_loss+0.5*loss_k

                errG.backward()
                optimizerG.step()

                global_step += 1
                if iteration % 100 == 0:
                    if rank == 0:
                        file_logger.info(
                            'epoch {} iteration{}, G Loss: {}, D Loss: {}, Rec Loss: {}'.format(epoch, iteration, errG.item(), errD.item(), Rec_loss))

            if not args.no_lr_decay:
                schedulerG.step()
                schedulerD.step()

            if rank == 0:
                if epoch % 10 == 0:
                    torchvision.utils.save_image(x_pos_sample, os.path.join(exp_path, 'xpos_epoch_{}.png'.format(epoch)),
                                                 normalize=True)
                idx = 0
                avg_psnr = 0.0
                avg_ssim = 0.0
                avg_lpips = 0.0
                for _, batch in enumerate(testloader, 1):
                    lr_data_down = batch[0]
                    real_data = batch[1]
                    real_data = to_range_1_1(real_data)
                    lr_data_down = to_range_1_1(lr_data_down)
                    lr_data_down = lr_data_down.to(device)
                    real_data = real_data.to(device)
                    
                    
                    x_t_1 = torch.randn_like(real_data).to(device)

                    fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, lr_data_down, T, args)
                    fake_sample = to_range_0_1(fake_sample)
                    real_data = to_range_0_1(real_data)
                    torchvision.utils.save_image(fake_sample, os.path.join(exp_path, 'sample_discrete_epoch_{}.png'.format(epoch)), normalize=True)
                    sr_img = util.tensor2img(fake_sample.float().cpu().squeeze())
                    gt_img = util.tensor2img(real_data.float().cpu().squeeze())

                    # calculate PSNR
                    crop_size = opt["scale"]
                    gt_img = gt_img / 255.0
                    sr_img = sr_img / 255.0
                    cropped_sr_img = sr_img[crop_size:-crop_size, crop_size:-crop_size]
                    cropped_gt_img = gt_img[crop_size:-crop_size, crop_size:-crop_size]

                    cropped_sr_img_y = bgr2ycbcr(cropped_sr_img, only_y=True)
                    cropped_gt_img_y = bgr2ycbcr(cropped_gt_img, only_y=True)


                    avg_psnr += util.calculate_psnr(
                        cropped_sr_img_y * 255, cropped_gt_img_y * 255
                    )
                    avg_ssim += util.calculate_ssim(cropped_sr_img_y * 255, cropped_gt_img_y * 255)
                    idx += 1

                avg_psnr = avg_psnr / idx
                avg_ssim = avg_ssim / idx
                file_logger.info(
                    "<epoch:{:3d},psnr: {:.6f},ssim:{:.6f}".format(
                        epoch, avg_psnr, avg_ssim
                    )
                )

                if args.save_content:
                    if epoch % args.save_content_every == 0:
                        print('Saving content.')
                        content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                                   'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                                   'schedulerG': schedulerG.state_dict(), 'netD_dict': netD.state_dict(),
                                   'optimizerD': optimizerD.state_dict(), 'schedulerD': schedulerD.state_dict()}

                        torch.save(content, os.path.join(exp_path, 'content.pth'))

                if epoch % args.save_ckpt_every == 0:
                    if args.use_ema:
                        optimizerG.swap_parameters_with_ema(store_params_in_ema=True)

                    torch.save(netG.state_dict(), os.path.join(exp_path, 'netG_{}.pth'.format(epoch)))
                    if args.use_ema:
                        optimizerG.swap_parameters_with_ema(store_params_in_ema=True)


def init_processes(rank, size, fn, args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '6060'
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args)
    dist.barrier()
    cleanup()


def cleanup():
    dist.destroy_process_group()


# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')

    parser.add_argument('--resume', action='store_true', default=False)

    parser.add_argument('--image_size', type=int, default=64,
                        help='size of image')
    parser.add_argument('--num_channels', type=int, default=3,
                        help='channel of image')
    parser.add_argument('--centered', action='store_false', default=True,
                        help='-1,1 scale')
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1,
                        help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20.,
                        help='beta_max for diffusion')

    parser.add_argument('--num_channels_dae', type=int, default=128,
                        help='number of initial channels in denosing model')
    parser.add_argument('--n_mlp', type=int, default=3,
                        help='number of mlp layers for z')
    parser.add_argument('--ch_mult', nargs='+', type=int,
                        help='channel multiplier')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                        help='number of resnet blocks per scale')
    parser.add_argument('--attn_resolutions', default=(16,),
                        help='resolution of applying attention')
    parser.add_argument('--dropout', type=float, default=0.,
                        help='drop-out rate')
    parser.add_argument('--resamp_with_conv', action='store_false', default=True,
                        help='always up/down sampling with conv')
    parser.add_argument('--conditional', action='store_false', default=True,
                        help='noise conditional')
    parser.add_argument('--fir', action='store_false', default=True,
                        help='FIR')
    parser.add_argument('--fir_kernel', default=[1, 3, 3, 1],
                        help='FIR kernel')
    parser.add_argument('--skip_rescale', action='store_false', default=True,
                        help='skip rescale')
    parser.add_argument('--resblock_type', default='biggan',
                        help='tyle of resnet block, choice in biggan and ddpm')
    parser.add_argument('--progressive', type=str, default='none', choices=['none', 'output_skip', 'residual'],
                        help='progressive type for output')
    parser.add_argument('--progressive_input', type=str, default='residual', choices=['none', 'input_skip', 'residual'],
                        help='progressive type for input')
    parser.add_argument('--progressive_combine', type=str, default='sum', choices=['sum', 'cat'],
                        help='progressive combine method.')

    parser.add_argument('--embedding_type', type=str, default='positional', choices=['positional', 'fourier'],
                        help='type of time embedding')
    parser.add_argument('--fourier_scale', type=float, default=16.,
                        help='scale of fourier transform')
    parser.add_argument('--not_use_tanh', action='store_true', default=False)

    # geenrator and training
    parser.add_argument('--exp', default='experiment_cifar_default', help='name of experiment')
    parser.add_argument('--dataset', default='DIV2K_Flickr2K', help='name of dataset')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=1200)
    parser.add_argument('--ngf', type=int, default=64)

    parser.add_argument('--lr_g', type=float, default=1.5e-4, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1e-4, help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam')
    parser.add_argument('--beta2', type=float, default=0.9,
                        help='beta2 for adam')
    parser.add_argument('--no_lr_decay', action='store_true', default=False)

    parser.add_argument('--use_ema', action='store_true', default=False,
                        help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')

    parser.add_argument('--r1_gamma', type=float, default=0.05, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=None,
                        help='lazy regulariation.')

    parser.add_argument('--save_content', action='store_true', default=False)
    parser.add_argument('--save_content_every', type=int, default=1, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=1, help='save ckpt every x epochs')

    ###ddp
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')
    ###新增模块####
    parser.add_argument("--data_dir", default='/tn/diffGAN-srd/', type=str,
                        help="Location to loading images")
    parser.add_argument('--patch_n', type=int, default=16)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument("--pca_matrix_path", default='/tn/FSRDiff-blur/pca_matrix/DCLS/pca_matrix.pth', type=str,
                        help="Location to pca_matrix_path")
    # parser.add_argument('--image_size', type=int, default=64)两个图像的大小了
    parser.add_argument('--training_batch_size', type=int, default=4)
    parser.add_argument('--sampling_batch_size', type=int, default=4)
    parser.add_argument('--hidden_size', type=int, default=64)  # UNet网络的初始通道数
    parser.add_argument("--rec_loss", action="store_true")
    parser.add_argument("--opt", type=str, help="Path to option YMAL file.")

    args = parser.parse_args()
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    opt = option.parse(args.opt, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, train, args))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        print('starting in debug mode')

        init_processes(0, size, train, args)
