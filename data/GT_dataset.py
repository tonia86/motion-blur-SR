import os
import random
import sys

import cv2
import lmdb
import numpy as np
import torch
import utils
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from utils.matlab_resize import imresize
from PIL import Image

try:
    sys.path.append("..")
    import data.util as util
except ImportError:
    pass


def tensor2img(img):
    img = np.round((img.permute(0, 2, 3, 1).cpu().numpy() + 1) * 127.5)
    img = img.clip(min=0, max=255).astype(np.uint8)
    return img


class GTDataset(data.Dataset):
    """
    Read LR (Low Quality, here is LR) and GT image pairs.
    The pair is ensured by 'sorted' function, so please check the name convention.
    """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.GT_paths = None
        self.GT_env = None  # environment for lmdb
        if self.opt['is_train']:
            self.GT_size = opt["GT_size"]
        self.data_aug_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20, resample=Image.BICUBIC),
#             transforms.RandomRotation(20, interpolation=Image.BICUBIC),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        ])
        self.to_tensor_norm = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
#         self.transforms = torchvision.transforms.Compose(
#             [torchvision.transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        # read image list from lmdb or image files
        if opt["data_type"] == "lmdb":
            self.GT_paths, self.GT_sizes = util.get_image_paths(
                opt["data_type"], opt["dataroot_GT"]
            )
        elif opt["data_type"] == "img":
            self.GT_paths = util.get_image_paths(
                opt["data_type"], opt["dataroot_GT"]
            )  # GT list
        else:
            print("Error: data_type is not matched in Dataset")
        assert self.GT_paths, "Error: GT paths are empty."

        self.random_scale_list = [1]

    def _init_lmdb(self):
        # https://github.com/chainer/chainermn/issues/129
        self.GT_env = lmdb.open(
            self.opt["dataroot_GT"],
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def __getitem__(self, index):
        if self.opt["data_type"] == "lmdb":
            if self.GT_env is None:
                self._init_lmdb()

        GT_path = None
        scale = self.opt["scale"]
        if self.opt['is_train']:
            GT_size = self.opt["GT_size"]

        # get GT image
        GT_path = self.GT_paths[index]
        if self.opt["data_type"] == "lmdb":
            resolution = [int(s) for s in self.GT_sizes[index].split("_")]
        else:
            resolution = None
        img_GT = util.read_img(
            self.GT_env, GT_path, resolution
        )  # return: Numpy float32, HWC, BGR, [0,1]

        # modcrop in the validation / test phase
        if self.opt["phase"] != "train":
            img_GT = util.modcrop(img_GT, scale)

        if self.opt["phase"] == "train":
            H, W, C = img_GT.shape
#             if self.data_augmentation and random.random() < 0.5:
#                 img_GT = self.data_augment(img_GT)
            rnd_h = random.randint(0, max(0, H - GT_size))
            rnd_w = random.randint(0, max(0, W - GT_size))
            img_GT = img_GT[rnd_h: rnd_h + GT_size, rnd_w: rnd_w + GT_size, :]
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
        img_LR = imresize(img_GT, 1/self.opt['scale'], method='bicubic')
        img_LR_UP = imresize(img_LR, self.opt['scale'])
        img_GT, img_LR, img_LR_UP = [self.to_tensor_norm(x).float() for x in [img_GT, img_LR, img_LR_UP]]
            # augmentation - flip, rotate
        #             img_GT = util.augment(
        #                 img_GT,
        #                 self.opt["use_flip"],
        #                 self.opt["use_rot"],
        #                 self.opt["mode"],
        #             )

        # change color space if necessary
        #         if self.opt["color"]:
        #             img_GT = util.channel_convert(img_GT.shape[2], self.opt["color"], [img_GT])[
        #                 0
        #             ]

        # BGR to RGB, HWC to CHW, numpy to tensor
        
        #         img_GT = torch.from_numpy(
        #             np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))
        #         ).float()

#         return {"GT": self.transforms(img_GT)}
        LR_path = GT_path
        return {'img_GT': img_GT, 'img_LR': img_LR, 'img_LR_UP': img_LR_UP, "LQ_path": LR_path, "GT_path": GT_path}

    def __len__(self):
        return len(self.GT_paths)
