from file import *
import SimpleITK as sitk
from image  import *
from visualizer import *
import glob
from tqdm import tqdm
import sys

import torch
import torchvision
import albumentations as A

def read_img(filename, msk_type):
        ''' read medical image
        Args:
            filename: the child data file directory
            msk_type: the type of label
        Return:
            image: the list of medical image
        '''
        img_sitks = sitk_get_image(filename, ignore_nonuniform=False)
        ds, _ = find_one_rs_file(filename)
        label_list = []
        img_list = []
        for img_sitk in img_sitks:
            label = rtstruct2stencil(img_sitk, ds, roi_names=[msk_type,'GTV'])[:,:,:,0].astype(np.float32)
            image = sitk.GetArrayFromImage(img_sitk)
            image = image_transform(image, clip_percent=[1, 99])
            print(label.shape)
            label_list.append(label)
            img_list.append(image)
        label_list = np.vstack(label_list)
        image = np.vstack(img_list)
        print(label_list.shape)
        return image, label_list



class MedDataSets3D(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=None):
        self.transforms = transform
        self.file_dir = glob.glob(
            os.path.join(img_dir,'*')
        )
    
    def __len__(self):
        return len(self.file_dir)
    
    def __getitem__(self, idx):
        img, msk = read_img(self.file_dir[idx], 'GTV-NP')
        sample = {"image": img, "label": msk}
        if self.transform:
            sample = self.transform(sample)
        return sample
