from re import L
from SimpleITK.SimpleITK import Mask
import sys
from file import *
import SimpleITK as sitk
from image  import *
from visualizer import *
import glob
from tqdm import tqdm
import sys



# img_sitks = sitk_get_image(r'E:\Process_Data\Bai^Li ping-RT180669\CT', ignore_nonuniform=False)
# ds, _ = find_one_rs_file(r'E:\Process_Data\Bai^Li ping-RT180669\CT')
# for img_sitk in img_sitks:
#     label = rtstruct2stencil(img_sitk, ds, roi_names=['GTV-NP']).astype(np.float32)
#     image = sitk.GetArrayFromImage(img_sitk)
#     image = image_transform(image, clip_percent=[1, 99])
#     for im in label:
#         print(im.shape)
#         cv2.imshow('im', im)
#         cv2.waitKey()



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
        label = rtstruct2stencil(img_sitk, ds, roi_names=[msk_type])[:,:,:,0].astype(np.float32)
        image = sitk.GetArrayFromImage(img_sitk)
        image = image_transform(image, clip_percent=[1, 99])
        label_list.append(label)
        img_list.append(image)
    label_list = np.vstack(label_list)[:,:,:,np.newaxis]
    image = np.vstack(img_list)[:,:,:,np.newaxis]
    return image, label_list

image, label_list = read_img(r'E:\Process_Data\Bai^Li ping-RT180669\CT','GTV-NP')
print(image.shape, label_list.shape)
for i in image:
    print(i.shape)
    cv2.imshow('im',i)
    cv2.waitKey()