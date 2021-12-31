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
    def __init__(self, size_W, size_H, size_D):
        self.size_W = size_W
        self.size_H = size_H
        self.size_D = size_D

    def get_range(self, ori_size: tuple):
        # ori_size: D,W,H,C
        rand_w_st = random.randint(self.size_W, ori_size[1] - self.size_W)
        rand_w_ed = rand_w_st - self.size_W
        rand_h_st = random.randint(self.size_H, ori_size[2] - self.size_H)
        rand_h_ed = rand_h_st - self.size_H
        rand_d_st = random.randint(self.size_D, ori_size[0] - self.size_D)
        rand_d_ed = rand_d_st - self.size_D
        return rand_w_st, rand_w_ed, rand_h_st, rand_h_ed, rand_d_st, rand_d_ed
    
    def crop_slices(self, label_list):
        ''' *crop the img and msk based on label*
        Args:
            label_list: the label: numpy.ndarry
        Return:
            label_slices_the(list): the up loop and end loop
        '''
        label_slices = [i for i,info in enumerate(label_list) if info.any()>0 ]
        length_label = len(label_slices)
        new_crop_slices = 32 - length_label
        label_slices_the = [label_slices[0]-int(new_crop_slices/2)-1, label_slices[-1] + int(new_crop_slices/2)+1]
        return label_slices_the

    def __call__(self, img, msk):
        rand_w_st, rand_w_ed, rand_h_st, rand_h_ed, rand_d_st, rand_d_ed = self.get_range(
            img.shape)
        tmp_img = torch.zeros(self.size_D, self.size_W, self.size_W,1)
        tmp_msk = tmp_img
        tmp_img[rand_d_st:rand_d_ed, rand_w_st:rand_w_ed, rand_h_st:rand_h_ed, :] = img[rand_d_st:rand_d_ed, rand_w_st:rand_w_ed, rand_h_st:rand_h_ed, :]
        tmp_msk[rand_d_st:rand_d_ed, rand_w_st:rand_w_ed, rand_h_st:rand_h_ed, :] = msk[rand_d_st:rand_d_ed, rand_w_st:rand_w_ed, rand_h_st:rand_h_ed, :]

        return tmp_img, tmp_msk
