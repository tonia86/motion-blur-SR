import os
import math
import numpy as np
import cv2
import glob
import torch
import lpips
import numpy as np
from math import ceil, floor
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
class Measure:
    def __init__(self, net='alex'):
        self.model = lpips.LPIPS(net=net)

    def measure(self, imgA, imgB, sr_scale):
        """

        Args:
            imgA: [C, H, W] uint8 or torch.FloatTensor [-1,1]
            imgB: [C, H, W] uint8 or torch.FloatTensor [-1,1]
            img_lr: [C, H, W] uint8  or torch.FloatTensor [-1,1]
            sr_scale:

        Returns: dict of metrics

        """
        if isinstance(imgA, torch.Tensor):
            imgA = np.round((imgA.cpu().numpy() + 1) * 127.5).clip(min=0, max=255).astype(np.uint8)
            imgB = np.round((imgB.cpu().numpy() + 1) * 127.5).clip(min=0, max=255).astype(np.uint8)
            # img_lr = np.round((img_lr.cpu().numpy() + 1) * 127.5).clip(min=0, max=255).astype(np.uint8)
        imgA = imgA.transpose(1, 2, 0)
        
        imgA_lr = imresize(imgA, 1 / sr_scale)
        imgB = imgB.transpose(1, 2, 0)
        img_lr = imresize(imgB, 1 / sr_scale)
        # img_lr = img_lr.transpose(1, 2, 0)
#        print(img_lr.shape)
#        print(imgA_lr.shape)
        psnr = self.psnr(imgA, imgB)
        ssim = self.ssim(imgA, imgB)
        lpips = self.lpips(imgA, imgB)
        lr_psnr = self.psnr(imgA_lr, img_lr)
        res = {'psnr': psnr, 'ssim': ssim, 'lpips': lpips, 'lr_psnr': lr_psnr}
        return {k: float(v) for k, v in res.items()}

    def lpips(self, imgA, imgB, model=None):
        device = next(self.model.parameters()).device
        tA = t(imgA).to(device)
        tB = t(imgB).to(device)
        dist01 = self.model.forward(tA, tB).item()
        return dist01

    def ssim(self, imgA, imgB):
        score, diff = ssim(imgA, imgB, full=True, multichannel=True, data_range=255)
        return score

    def psnr(self, imgA, imgB):
        return psnr(imgA, imgB, data_range=255)

def deriveSizeFromScale(img_shape, scale):
    output_shape = []
    for k in range(2):
        output_shape.append(int(ceil(scale[k] * img_shape[k])))
    return output_shape


def deriveScaleFromSize(img_shape_in, img_shape_out):
    scale = []
    for k in range(2):
        scale.append(1.0 * img_shape_out[k] / img_shape_in[k])
    return scale


def triangle(x):
    x = np.array(x).astype(np.float64)
    lessthanzero = np.logical_and((x >= -1), x < 0)
    greaterthanzero = np.logical_and((x <= 1), x >= 0)
    f = np.multiply((x + 1), lessthanzero) + np.multiply((1 - x), greaterthanzero)
    return f


def cubic(x):
    x = np.array(x).astype(np.float64)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    f = np.multiply(1.5 * absx3 - 2.5 * absx2 + 1, absx <= 1) + np.multiply(-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2,
                                                                            (1 < absx) & (absx <= 2))
    return f


def contributions(in_length, out_length, scale, kernel, k_width):
    if scale < 1:
        h = lambda x: scale * kernel(scale * x)
        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x = np.arange(1, out_length + 1).astype(np.float64)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = int(ceil(kernel_width)) + 2
    ind = np.expand_dims(left, axis=1) + np.arange(P) - 1  # -1 because indexing from 0
    indices = ind.astype(np.int32)
    weights = h(np.expand_dims(u, axis=1) - indices - 1)  # -1 because indexing from 0
    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]
    return weights, indices


def imresizemex(inimg, weights, indices, dim):
    in_shape = inimg.shape
    w_shape = weights.shape
    out_shape = list(in_shape)
    out_shape[dim] = w_shape[0]
    outimg = np.zeros(out_shape)
    if dim == 0:
        for i_img in range(in_shape[1]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[ind, i_img].astype(np.float64)
                outimg[i_w, i_img] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)
    elif dim == 1:
        for i_img in range(in_shape[0]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[i_img, ind].astype(np.float64)
                outimg[i_img, i_w] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg


def imresizevec(inimg, weights, indices, dim):
    wshape = weights.shape
    if dim == 0:
        weights = weights.reshape((wshape[0], wshape[2], 1, 1))
        outimg = np.sum(weights * ((inimg[indices].squeeze(axis=1)).astype(np.float64)), axis=1)
    elif dim == 1:
        weights = weights.reshape((1, wshape[0], wshape[2], 1))
        outimg = np.sum(weights * ((inimg[:, indices].squeeze(axis=2)).astype(np.float64)), axis=2)
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg


def resizeAlongDim(A, dim, weights, indices, mode="vec"):
    if mode == "org":
        out = imresizemex(A, weights, indices, dim)
    else:
        out = imresizevec(A, weights, indices, dim)
    return out


def imresize(I, scale=None, method='bicubic', sizes=None, mode="vec"):
    if method == 'bicubic':
        kernel = cubic
    elif method == 'bilinear':
        kernel = triangle
    else:
        print('Error: Unidentified method supplied')

    kernel_width = 4.0
    # Fill scale and output_size
    if scale is not None:
        scale = float(scale)
        scale = [scale, scale]
        output_size = deriveSizeFromScale(I.shape, scale)
    elif sizes is not None:
        scale = deriveScaleFromSize(I.shape, sizes)
        output_size = list(sizes)
    else:
        print('Error: scalar_scale OR output_shape should be defined!')
        return
    scale_np = np.array(scale)
    order = np.argsort(scale_np)
    weights = []
    indices = []
    for k in range(2):
        w, ind = contributions(I.shape[k], output_size[k], scale[k], kernel, kernel_width)
        weights.append(w)
        indices.append(ind)
    B = np.copy(I)
    flag2D = False
    if B.ndim == 2:
        B = np.expand_dims(B, axis=2)
        flag2D = True
    for k in range(2):
        dim = order[k]
        B = resizeAlongDim(B, dim, weights[dim], indices[dim], mode)
    if flag2D:
        B = np.squeeze(B, axis=2)
    return B

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
Measure = Measure()
folder_GT = '/tn/diffGAN-srd/data/dataset/research/setting1/DIV2K_valid_HR/DIV2K_valid_HR'
folder_Gen = '/tn/FSRDiff/Result/residual001/DIV2KRK/'

crop_border = 0
suffix = ''  # suffix for Gen images
test_Y = False
# True: test Y channel only; False: test RGB channels

PSNR_all = []
LR_PSNR_all = []
SSIM_all = []
LPIPS_all = []
img_list = sorted(glob.glob(folder_GT + '/*'))

if test_Y:
    print('Testing Y channel.')
else:
    print('Testing RGB channels.')

for i, img_path in enumerate(img_list):
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    im_GT = cv2.imread(img_path)
    im_GT = im_GT[:, :, [2, 1, 0]]
    im_GT = np.transpose(im_GT, (2, 0, 1))
#    print(im_GT.shape)
    # print(im_GT)
    im_Gen = cv2.imread(os.path.join(folder_Gen, base_name + suffix + '.png'))
    im_Gen = im_Gen[:, :, [2, 1, 0]]
    im_Gen = np.transpose(im_Gen, (2, 0, 1))
    # im_Gen = im_Gen[:,:,::-1].transpose((2,0,1))
    # im_GT_lr = imresize(im_GT, 1 / 4)
#    print(im_Gen.shape)
    s =Measure.measure(imgB=im_GT, imgA=im_Gen,sr_scale=4)
    print(s)
