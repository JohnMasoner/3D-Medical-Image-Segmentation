from .file import *
import SimpleITK as sitk
from .image  import *
from .visualizer import *
import glob

import torch
from torchvision import transforms
from .transforms import RandomCrop

class MedDataSets3D(torch.utils.data.Dataset):
    def __init__(self, img_dir, transform=False,length = (None,None)):
        self.transform = transform
        self.file_dir = glob.glob(
            os.path.join(img_dir,'*')
        )[length[0]:length[-1]]
    
    def __getitem__(self, idx):
        img, msk = self.read_img(self.file_dir[idx], 'GTV-NP')
        # tsf = self.transf(image=img.astype('uint8'), mask=msk)
        # img, msk = tsf['image'], tsf['mask']
        # sample = {"image": img.astype(np.float32), "label": msk.astype(np.float32)}
        # trans = RandomCrop(512,512,32)
        # img, msk = trans(img, msk)
        # print(img.shape)
        sample = {"image": torch.transpose(img,3,0), "label": torch.transpose(msk,3,0)}
        if self.transform:
            sample = self.transform(sample)
        # sample = {"image": torch.transpose(sample['image'],3,0), "label": torch.transpose(sample['label'],3,0)}
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
        # bc the img_sitks is nonuniform ,the img_sitk is a list for the image info
        for img_sitk in img_sitks:
            label = rtstruct2stencil(img_sitk, ds, roi_names=msk_type)[:,:,:,0].astype(np.float32)
            image = sitk.GetArrayFromImage(img_sitk)
            image = image_transform(image, clip_percent=[1, 99])
            label_list.append(label)
            img_list.append(image)
        # get any numpy.ndarry have to concat the ndarry
        label_list = np.vstack(label_list)[:,:,:,np.newaxis]
        # the data shape is D,W,H,C
        image = np.vstack(img_list)[:,:,:,np.newaxis]
        image = torch.from_numpy(image)
        label_list = torch.from_numpy(label_list)
        return image, label_list
    
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

if __name__ == '__main__':
    train_data = MedDataSets3D('E:/Y054/Process_Data')
    train_dl = torch.utils.data.DataLoader(train_data, 1, True, num_workers=1)
    for i, sample in enumerate(train_dl):
        print(i,type(sample))