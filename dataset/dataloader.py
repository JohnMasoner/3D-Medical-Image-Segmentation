from file import *
import SimpleITK as sitk
from image  import *
from visualizer import *
import glob
from tqdm import tqdm
import sys

import torch
import torchio

from transforms import RandomCrop

class MedDataSets3D(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=False):
        self.transform = transform
        self.file_dir = glob.glob(
            os.path.join(img_dir,'*/CT')
        )[:2]
    
    def __getitem__(self, idx):
        img, msk = self.read_img(self.file_dir[idx], 'GTV-NP')
        # tsf = self.transf(image=img.astype('uint8'), mask=msk)
        # img, msk = tsf['image'], tsf['mask']
        # sample = {"image": img.astype(np.float32), "label": msk.astype(np.float32)}
        trans = RandomCrop(512,512,32)
        img, msk = trans(img, msk)
        print(img.shape)
        sample = {"image": img, "label": msk}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.file_dir)
    
    def read_img(self, filename, msk_type):
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
            label = rtstruct2stencil(img_sitk, ds, roi_names=msk_type)[:,:,:,0].astype(np.float32)
            image = sitk.GetArrayFromImage(img_sitk)
            image = image_transform(image, clip_percent=[1, 99])
            label_list.append(label)
            img_list.append(image)
        label_list = np.vstack(label_list)[:,:,:,np.newaxis]
        image = np.vstack(img_list)[:,:,:,np.newaxis]
        print(label_list.shape, image.shape)
        image = torch.from_numpy(image)
        label_list = torch.from_numpy(label_list)
        return image, label_list

if __name__ == '__main__':
    train_data = MedDataSets3D('E:/Process_Data')
    train_dl = torch.utils.data.DataLoader(train_data, 1, True, num_workers=1)
    for i, sample in enumerate(train_dl):
        print(i,type(sample))