# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import argparse
import torch
import numpy as np
import logging
import os.path
import sys
import time
from torch import nn
import utils as util
from data.util import imresize
import os
from collections import OrderedDict
from data import create_dataloader, create_dataset
from data.dataset_hf5 import DataSet, DataValSet
from torch.utils.data import DataLoader
from data.util import bgr2ycbcr
import torchvision
from score_sde.models.ncsnpp_blur_sr_gfn import Unet
from pytorch_fid.fid_score import calculate_fid_given_paths
import options as option
import lpips
from thop import profile
from time import time
print(torch.cuda.device_count())
# %% Diffusion coefficients
def var_func_vp(t, beta_min, beta_max):
    log_mean_coeff = -0.25 * t ** 2 * (beta_max - beta_min) - 0.5 * t * beta_min
    var = 1. - torch.exp(2. * log_mean_coeff)
    return var


def var_func_geometric(t, beta_min, beta_max):
    return beta_min * ((beta_max / beta_min) ** t)


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
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
    n_timestep = args.num_timesteps
    beta_min = args.beta_min
    beta_max = args.beta_max
    eps_small = 1e-3

    t = np.arange(0, n_timestep + 1, dtype=np.float64)
    t = t / n_timestep
    t = torch.from_numpy(t) * (1. - eps_small) + eps_small

    if args.use_geometric:
        var = var_func_geometric(t, beta_min, beta_max)
    else:
        var = var_func_vp(t, beta_min, beta_max)
    alpha_bars = 1.0 - var
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]

    first = torch.tensor(1e-8)
    betas = torch.cat((first[None], betas)).to(device)
    betas = betas.type(torch.float32)
    sigmas = betas ** 0.5
    a_s = torch.sqrt(1 - betas)
    return sigmas, a_s, betas


# %% posterior sampling
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

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.rsqrt(self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        self.posterior_mean_coef2 = (
                    (1 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1 - self.alphas_cumprod))

        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))


def sample_posterior(coefficients, x_0, x_t, t):
    def q_posterior(x_0, x_t, t):
#         print(t)
#         print(type(extract(coefficients.sqrt_recipm1_alphas_cumprod, t, x_t.shape)))
#         print(extract(coefficients.posterior_mean_coef2, t, x_t.shape))
#         print(x_0.size())
#         x_00 = (
#                 extract(coefficients.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
#                 extract(coefficients.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * x_0
#         )
#         print('-----------------------')
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

# x_tp1.detach(), t, lr_data_down, lr_data
def sample_from_model(coefficients, generator, n_time, x_init, lr_data_down, T, opt):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)

            t_time = t
            latent_z = torch.randn(x.size(0), opt.nz, device=x.device)
            x_0, _ = generator(x, t_time, lr_data_down)
#             print('##################################')
#             print(x_0.size())
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()

    return x

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

def b_Bicubic(variable, scale):
    B, C, H, W = variable.size()
    H_new = int(H / scale)
    W_new = int(W / scale)
    tensor_v = variable.view((B, C, H, W))
    re_tensor = imresize(tensor_v, 1 / scale)
    return re_tensor

# %%
def sample_and_test(args):
    util.mkdirs(
        (
            path
            for key, path in opt["path"].items()
            if not key == "experiments_root"
               and "pretrain_model" not in key
               and "resume" not in key
        )
    )

    os.system("rm ./result")
    os.symlink(os.path.join(opt["path"]["results_root"], ".."), "./result")

    util.setup_logger(
        "base",
        opt["path"]["log"],
        "test_" + opt["name"],
        level=logging.INFO,
        screen=True,
        tofile=True,
    )
    logger = logging.getLogger("base")
    logger.info(option.dict2str(opt))

    #### Create test dataset and dataloader
#     for phase, dataset_opt in sorted(opt["datasets"].items()):
#         test_set = create_dataset(dataset_opt)
#         test_loader = torch.utils.data.DataLoader(
#             test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
#         )
    root_val_dir = '/tn/Data/Validation_4/'# #----------Validation path
    test_loader = DataLoader(DataValSet(root_val_dir), batch_size=1, shuffle=False, pin_memory=False) 
                
#     test_loaders = []
#     for phase, dataset_opt in sorted(opt["datasets"].items()):
#         test_set = create_dataset(dataset_opt)
#         test_loader = create_dataloader(test_set, dataset_opt)
#         logger.info(
#             "Number of test images in [{:s}]: {:d}".format(
#                 dataset_opt["name"], len(test_set)
#             )
#         )
#         test_loaders.append(test_loader)

    dim_mults = args.ch_mult
    torch.manual_seed(42)
    device = 'cuda:1'
    to_range_1_1 = lambda x: 2.*x-1.
    to_range_0_1 = lambda x: (x + 1.) / 2.  # 将-1，1的值转换到0，1
    netG = Unet(opt, 64, out_dim=3, cond_dim=32, dim_mults=dim_mults).to(device)
#     t = torch.randint(0, 15, (1,),
#                                   device=device)
#     print(t)
#     flops, params = profile(netG, inputs=(torch.rand(1, 3, 224, 224).to(device), t.to(device), torch.rand(1, 3, 56, 56).to(device)), verbose=False)
#     print('FLOPs = ' + str(flops / 1000 ** 2) + 'M')
#     print('Params = ' + str(params / 1000 ** 2) + 'M')
    ckpt = torch.load('./saved_info/dd_gan/{}/{}/netG_{}.pth'.format('DIV2K_Flickr2K', args.exp, args.epoch_id),map_location = device)
    for key in list(ckpt.keys()):
        ckpt[key[7:]] = ckpt.pop(key)
    netG.load_state_dict(ckpt)
    netG.eval()
    
    T = get_time_schedule(args, device)

    pos_coeff = Posterior_Coefficients(args, device)

    iters_needed = 50000 // args.batch_size

    save_dir = "./generated_samples/{}".format(args.dataset)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for _, batch in enumerate(test_loader, 1):
        lr_data_down = torch.rand(1, 3, 64, 64).to(device)
        real_data = torch.rand(1, 3, 256, 256).to(device)
        img_name = batch[2]
#         real_data = to_range_1_1(real_data)
#         lr_data_down = to_range_1_1(lr_data_down)
        lr_data_down = lr_data_down.to(device)
        real_data = real_data.to(device)
        idx = 0
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_lpips = 0.0

        gt_img = util.tensor2img(real_data.float().cpu().squeeze())
        x_t_1 = torch.randn_like(real_data).to(device)
        start= time()
        fake_sample = sample_from_model(pos_coeff, netG, args.num_timesteps, x_t_1, lr_data_down, T, args)
        end = time()
        print(end-start)
#         fake_sample = to_range_0_1(fake_sample)


        suffix = opt["suffix"]
        dataset_dir = '/tn/work3/FSRDiff-blur/Result/blur/t/'
        if suffix:
            save_img_path = os.path.join(dataset_dir, str(img_name[0]) + suffix)
        else:
            save_img_path = os.path.join(dataset_dir, str(img_name[0]))
        torchvision.utils.save_image(fake_sample,
                                             os.path.join(save_img_path),
                                             normalize=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('ddgan parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
    parser.add_argument('--compute_fid', action='store_true', default=False,
                        help='whether or not compute FID')
    parser.add_argument('--epoch_id', type=int, default=1000)
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
    parser.add_argument('--real_img_dir', default='./pytorch_fid/cifar10_train_stat.npy',
                        help='directory to real images for FID computation')

    parser.add_argument('--dataset', default='cifar10', help='name of dataset')
    parser.add_argument('--image_size', type=int, default=32,
                        help='size of image')

    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--num_timesteps', type=int, default=4)

    parser.add_argument('--z_emb_dim', type=int, default=256)
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=1, help='sample generating batch size')
    parser.add_argument("--data_dir", default='/tn/diffusionGAN-srdiff/', type=str,
                        help="Location to loading images")
    parser.add_argument('--patch_n', type=int, default=1)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument("--pca_matrix_path", default='/tn/diffusionGAN-srdiff/pca_matrix.pth', type=str,
                        help="Location to pca_matrix_path")
    # parser.add_argument('--image_size', type=int, default=64)两个图像的大小了
    parser.add_argument('--training_batch_size', type=int, default=4)
    parser.add_argument('--sampling_batch_size', type=int, default=4)
    parser.add_argument('--hidden_size', type=int, default=64)  # UNet网络的初始通道数
    parser.add_argument("--opt", type=str, help="Path to option YMAL file.")


    args = parser.parse_args()
    opt = option.parse(parser.parse_args().opt, is_train=False)
    sample_and_test(args)


