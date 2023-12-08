import os
import math
import numpy as np
import cv2
import glob
import torch
import lpips
import numpy as np
from collections import OrderedDict
import torchvision
import torchvision.transforms as transforms
from pytorch_fid import fid_score

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
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))

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

def ssim(img1, img2):
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    """calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    """
    if not img1.shape == img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError("Wrong input image dimensions.")


folder_GT = '/tn/Data/Validation_4/HR/' #'/tn/Data/DIV2K_Gaussian8/HR/x4/'
folder_Gen = '/tn/work3/FSRDiff-blur/Result/blur_sr_t1_18/'

# crop_border = 0
suffix = ''  # suffix for Gen images
test_Y = False
# True: test Y channel only; False: test RGB channels
test_results = OrderedDict()
test_results["psnr"] = []
test_results["ssim"] = []
test_results["psnr_y"] = []
test_results["ssim_y"] = []
test_times = []
test_results["LPIPS"] = []
test_results["FID"] = []
img_list = sorted(glob.glob(folder_GT + '/*'))

if test_Y:
    print('Testing Y channel.')
else:
    print('Testing RGB channels.')


for i, img_path in enumerate(img_list):
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    gt_img = cv2.imread(img_path)
#     gt_img = gt_img[:, :, [2, 1, 0]]
#     gt_img = np.transpose(gt_img, (2, 0, 1))
#     C,H,W = gt_img.shape
    #print(im_GT.shape)
#     print(img_name)
#     print(gt_img)
#     suffix = '_SRDiff_sr_diffir_deblur'
    sr_img = cv2.imread(os.path.join(folder_Gen, img_name + suffix + '.png'))
#     sr_img = sr_img[:, :, [2, 1, 0]]
#     sr_img = np.transpose(sr_img, (2, 0, 1))[:,:H,:W]
    gt_img = gt_img / 255.0
    sr_img = sr_img / 255.0
    crop_border = 4
    if crop_border == 0:
        cropped_sr_img = sr_img
        cropped_gt_img = gt_img
    else:
        cropped_sr_img = sr_img[
                         crop_border:-crop_border, crop_border:-crop_border
                         ]
        cropped_gt_img = gt_img[
                         crop_border:-crop_border, crop_border:-crop_border
                         ]
    psnr = calculate_psnr(cropped_sr_img * 255, cropped_gt_img * 255)
    SSIM = calculate_ssim(cropped_sr_img * 255, cropped_gt_img * 255)
    LPIPS = Lpips(cropped_sr_img * 255, cropped_gt_img * 255)
    test_results["psnr"].append(psnr)
    test_results["ssim"].append(SSIM)
    test_results["LPIPS"].append(LPIPS)

    if len(gt_img.shape) == 3:
        if gt_img.shape[2] == 3:  # RGB image
            sr_img_y = bgr2ycbcr(sr_img, only_y=True)
            gt_img_y = bgr2ycbcr(gt_img, only_y=True)
            if crop_border == 0:
                cropped_sr_img_y = sr_img_y
                cropped_gt_img_y = gt_img_y
            else:
                cropped_sr_img_y = sr_img_y[
                                   crop_border:-crop_border, crop_border:-crop_border
                                   ]
                cropped_gt_img_y = gt_img_y[
                                   crop_border:-crop_border, crop_border:-crop_border
                                   ]
            psnr_y = calculate_psnr(
                cropped_sr_img_y * 255, cropped_gt_img_y * 255
            )
            ssim_y = calculate_ssim(
                cropped_sr_img_y * 255, cropped_gt_img_y * 255
            )

            test_results["psnr_y"].append(psnr_y)
            test_results["ssim_y"].append(ssim_y)

            print(
                "img:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}; LPIPS: {:.6f}; PSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}.".format(
                    img_name, psnr, SSIM, LPIPS, psnr_y, ssim_y
                )
            )
    else:
        print(
            "img:{:15s} - PSNR: {:.6f} dB; SSIM: {:.6f}; LPIPS: {:.6f}.".format(
                img_name, psnr, SSIM, LPIPS
            )
        )

        test_results["psnr_y"].append(psnr)
        print('no runer')
        test_results["ssim_y"].append(SSIM)
else:
    print(img_name)

ave_psnr = sum(test_results["psnr"]) / len(test_results["psnr"])
ave_ssim = sum(test_results["ssim"]) / len(test_results["ssim"])
ave_lpips = sum(test_results["LPIPS"]) / len(test_results["LPIPS"])
print(
    "----Average PSNR/SSIM results for DIV2KRK----\n\tPSNR: {:.6f} dB; SSIM: {:.6f}; LPIPS: {:.6f}\n".format(
         ave_psnr, ave_ssim, ave_lpips
    )
)
if test_results["psnr_y"] and test_results["ssim_y"]:
    ave_psnr_y = sum(test_results["psnr_y"]) / len(test_results["psnr_y"])
    ave_ssim_y = sum(test_results["ssim_y"]) / len(test_results["ssim_y"])
    print(
        "----Y channel, average PSNR/SSIM----\n\tPSNR_Y: {:.6f} dB; SSIM_Y: {:.6f}\n".format(
            ave_psnr_y, ave_ssim_y
        )
    )

print(f"average test time: {np.mean(test_times):.4f}")
