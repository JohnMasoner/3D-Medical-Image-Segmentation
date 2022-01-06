from torch.utils.data import Dataset
import torch
import numpy as np
import random
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import normalize

random.seed(1)


class RandomFlip_LR:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[0] <= self.prob:
            img = img.flip(2)
        return img

    def __call__(self, img, mask):
        prob = (random.uniform(0, 1), random.uniform(0, 1))
        return self._flip(img, prob), self._flip(mask, prob)


class RandomFlip_UD:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob[1] <= self.prob:
            img = img.flip(3)
        return img

    def __call__(self, img, mask):
        prob = (random.uniform(0, 1), random.uniform(0, 1))
        return self._flip(img, prob), self._flip(mask, prob)


class RandomRotate:
    def __init__(self, max_cnt=3):
        self.max_cnt = max_cnt

    def _rotate(self, img, cnt):
        img = torch.rot90(img, cnt, [1, 2])
        return img

    def __call__(self, img, mask):
        cnt = random.randint(0, self.max_cnt)
        return self._rotate(img, cnt), self._rotate(mask, cnt)


class RandomCrop:
    def __init__(self, rand_crop_size, n=1):
        self.size_W = rand_crop_size[0]
        self.size_H = rand_crop_size[1]
        self.size_D = rand_crop_size[2]
        self.n = n

    def get_range(self, ori_size: tuple):
        # ori_size: D,W,H,C
        rand_d_st = random.randint(self.size_D, ori_size[0])
        rand_d_ed = rand_d_st - self.size_D
        rand_w_st = random.randint(self.size_W, ori_size[1])
        rand_w_ed = rand_w_st - self.size_W
        rand_h_st = random.randint(self.size_H, ori_size[2])
        rand_h_ed = rand_h_st - self.size_H
        return rand_w_st, rand_w_ed, rand_h_st, rand_h_ed, rand_d_st, rand_d_ed

    def crop(self,img, msk, start: int, end: int, left: int, right: int, top: int, bottom: int):
        '''Crop the given image at specified location and output size.
        Args:
            img (Tensor): Image to be cropped. (0,0) denotes the top left corner of the image.
            img (Tensor): Image to be cropped. (0,0) denotes the top left corner of the mask.
            start (int): the first layer of the crop box.
            end (int): the last layer of the crop box.
            left (int): Vertical component of the top left corner of the crop box.
            right (int): Horizontal component of the top left corner of the crop box.
            top (int): Height of the crop box.
            bottom (int): Width of the crop box.
        Returns:
            crop_img (Tensor): Cropped image.
        '''
        crop_img = torch.zeros(self.size_D, self.size_W, self.size_W,1)
        crop_msk = crop_img
        crop_img = img[start:end, left:right, top:bottom, :]
        crop_msk = msk[start:end, left:right, top:bottom, :]
        return crop_img, crop_msk

    def __call__(self,data):
        img, msk = data['image'], data['label']

        rand_w_ed, rand_w_st, rand_h_ed, rand_h_st, rand_d_ed, rand_d_st = self.get_range(
            img.shape)
        img0, msk0 = self.crop(img, msk, rand_d_st, rand_d_ed, rand_w_st, rand_w_ed, rand_h_st, rand_h_ed)

        rand_w_ed, rand_w_st, rand_h_ed, rand_h_st, rand_d_ed, rand_d_st = self.get_range(
            img.shape)
        img1, msk1 = self.crop(img, msk, rand_d_st, rand_d_ed, rand_w_st, rand_w_ed, rand_h_st, rand_h_ed)

        rand_w_ed, rand_w_st, rand_h_ed, rand_h_st, rand_d_ed, rand_d_st = self.get_range(
            img.shape)
        img2, msk2 = self.crop(img, msk, rand_d_st, rand_d_ed, rand_w_st, rand_w_ed, rand_h_st, rand_h_ed)

        rand_w_ed, rand_w_st, rand_h_ed, rand_h_st, rand_d_ed, rand_d_st = self.get_range(
            img.shape)
        img3, msk3 = self.crop(img, msk, rand_d_st, rand_d_ed, rand_w_st, rand_w_ed, rand_h_st, rand_h_ed)

        return {
            'image': (img0, img1, img2, img3),
            'label': (msk0, msk1, msk2, msk3)
        }