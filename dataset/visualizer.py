import logging
import math
import os
from io import BytesIO
from typing import Union

import cv2
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image
from tensorboard.compat.proto.summary_pb2 import Summary
from torch.utils.tensorboard import SummaryWriter

from . import file as mtfile
from . import image as mtimage
# import file as mtfile
# import image as mtimage


def get_color(c, x, max_):
    colors_ = [[1, 0, 1], [0, 0, 1], [0, 1, 1],
               [0, 1, 0], [1, 1, 0], [1, 0, 0]]

    ratio = (float(x)/max_)*5
    i = math.floor(ratio)
    j = math.ceil(ratio)
    ratio -= i

    r = (1-ratio) * colors_[int(i)][int(c)] + ratio*colors_[int(j)][int(c)]

    return r


def generate_color_panel(N):
    color_panel = []
    for i in range(N):
        offset = i*123457 % N
        blue = get_color(2, offset, N)
        green = get_color(1, offset, N)
        red = get_color(0, offset, N)
        red = min(red*255, 255)
        green = min(green*255, 255)
        blue = min(blue*255, 255)
        color_panel.append((blue, green, red))
    return color_panel


# Input param:
#     [ims]: binary label image(value range is 0 or 1) with shape [D, H, W, C], D is slices, C is class
#     [image]: CT or MR image(value range is 0.0~1.0)
# Function:
#     if [image] is None, return colored label mask based on [ims].
#     if [image] is not None, return [image] covered with colored label mask
def colorization(ims, image=None):
    D, H, W, C = ims.shape
    ims = ims.astype(np.float32)

    panel = generate_color_panel(C)
    color_map = np.zeros((D, H, W, 3), np.float32)
    for i in range(C):
        mask = ims[..., i]
        alpha = (mask > 0).astype(np.float32)
        alpha = alpha[np.newaxis, ...]
        alpha = np.concatenate([alpha, alpha, alpha])
        alpha = alpha.transpose(1, 2, 3, 0)
        color_mask = np.zeros((D, H, W, 3), np.float32)
        for idx in range(3):
            color_mask[..., idx] = mask*panel[i][idx]
        foreground = cv2.multiply(alpha, color_mask)
        background = cv2.multiply(1.0 - alpha, color_map)
        color_map = cv2.add(foreground, background)

    color_map = np.clip(color_map, 0, 255)/255.0
    if D == 1:
        ims = [color_map[0]]
    else:
        ims = [color_map[i] for i in range(D)]
    if image is not None:
        nims = []
        for img, lab in zip(image, ims):
            tmp = np.array([img for _ in range(3)]).transpose(1, 2, 0)
            mask_t = (lab[..., 0] > 0) | (lab[..., 1] > 0) | (lab[..., 2] > 0)
            mask = np.array([mask_t for _ in range(3)]).transpose(1, 2, 0)
            canvas = np.where(mask > 0, lab, tmp)
            nims.append(canvas)
        ims = nims
    return ims


def draw_and_show(image, label, cover=True, crop=None, out_dir=None):
    if label is not None:
        logging.debug('start colorization')
        if cover:
            label = colorization(label, image=image)
        else:
            label = colorization(label)
        logging.debug('\tcomplete colorization')
        for idx, (img, lab) in enumerate(zip(image, label)):
            if crop is not None:
                h, w = img.shape[:2]
                xl, yl = int(w*crop), int(h*crop)
                xr, yr = w-xl, h-yl
                img = img[yl:yr, xl:xr]
                lab = lab[yl:yr, xl:xr]
            if out_dir is not None:
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                if not cover:
                    cv2.imwrite(os.path.join(out_dir, '{}_img.png'.format(
                        idx)), (img*255).astype(np.uint8))
                cv2.imwrite(os.path.join(out_dir, '{}_lab.png'.format(
                    idx)), (lab*255).astype(np.uint8))
            else:
                if not cover:
                    cv2.imshow('img', img)
                cv2.imshow('lab', lab)
                cv2.waitKey()
    else:
        for idx, img in enumerate(image):
            if out_dir is not None:
                if not os.path.exists(out_dir):
                    os.mkdir(out_dir)
                cv2.imwrite(os.path.join(out_dir, '{}_img.png'.format(
                    idx)), (img*255).astype(np.uint8))
            else:
                cv2.imshow('img', img)
                cv2.waitKey()


def visualize_dicom(patient_dir, roi_names,
                    clip_value=None, clip_percent=None, fill_contour=True, inds_for_smooth=[], do_reorientation=True,
                    out_dir=None, crop=None, cover=True):
    # read image
    logging.debug('start read image')
    slices = mtfile.sitk_get_image(patient_dir)
    if do_reorientation:
        slices = mtimage.reorientation(slices)
    image = sitk.GetArrayFromImage(slices)
    image = mtimage.image_transform(image, clip_value=clip_value,
                            clip_percent=clip_percent)
    logging.debug('\tcomplete read image')

    # read label
    logging.debug('start read label')
    label = None
    rt_dcm, _ = mtfile.find_one_rs_file(patient_dir)
    if rt_dcm is not None and roi_names is not None:
        roi_names = [roi.lower() for roi in roi_names]
        label = mtfile.rtstruct2stencilV1(
            slices, rt_dcm, roi_names, fill_contour=fill_contour, inds_for_smooth=inds_for_smooth)
    logging.debug('\tcomplete read label')

    draw_and_show(image, label, cover=cover, crop=crop, out_dir=out_dir)


def convert_dense_to_onehot(label):
    '''
        The number of classes is equal to the number of unique values 0
    '''
    classes = np.unique(label)
    classes = classes[classes > 0]
    classes = np.sort(classes)
    newlabel = np.zeros(list(label.shape)+[len(classes)], dtype=np.uint8)
    for idx, c in enumerate(classes):
        newlabel[..., idx] = (label == c)
    return newlabel


def visualize_nii(image_path, label_path=None,
                  clip_value=None, clip_percent=None,
                  out_dir=None, crop=None, cover=True):
    # read image
    logging.debug('start read image')
    img_nib = nib.load(image_path)
    img_nib = mtimage.reorientation_nib(img_nib)
    image = img_nib.get_fdata()
    image = mtimage.image_transform(image, clip_value=clip_value,
                            clip_percent=clip_percent)
    logging.debug('\tcomplete read image')

    # read label
    logging.debug('start read label')
    label = None
    if label_path is not None:
        lab_nib = nib.load(label_path)
        lab_nib = mtimage.reorientation_nib(lab_nib)
        label = lab_nib.get_fdata()
        assert image.shape == label.shape, 'shapes of image and label must be same'
        label = convert_dense_to_onehot(label)
    logging.debug('\tcomplete read label')

    draw_and_show(image, label, cover=cover, crop=crop, out_dir=out_dir)


def _gifims_to_summary(tag: str, ims: list, colorspace: int = 1) -> Summary:
    """
        convert ims to summary. this func is used in add_image_to_summary in torch tensorboard
        ims: list of numpy array (HW or HWC)
    """

    img, *img_rest = [Image.fromarray(im) for im in ims]
    with BytesIO() as f:
        img.save(f, format='GIF', save_all=True,
                 append_images=img_rest, duration=200, loop=0)
        img_str = f.getvalue()
    summary_image_str = Summary.Image(
        height=10, width=10, colorspace=colorspace, encoded_image_string=img_str)
    image_summary = Summary.Value(tag=tag, image=summary_image_str)
    return Summary(value=[image_summary])


def add_image_to_summary(
    data: Union[torch.Tensor, np.ndarray],
    step: int,
    writer: SummaryWriter,
    index: int = 0,
    max_channels: int = 1,
    max_frames: int = 10,
    tag: str = "output",
    do_coloring: bool = False,
) -> None:
    """Plot 2D or 3D image on the TensorBoard, 3D image will be converted to GIF image.

    Note:
        Plot 3D or 2D image(with more than 3 channels) as separate images.

    Args:
        data: target data to be plotted as image on the TensorBoard.
            The data is expected to have 'NCHW[D]' dimensions, and only plot the first in the batch.
        step: current step to plot in a chart.
        writer: specify TensorBoard SummaryWriter to plot the image.
        index: plot which element in the input data batch, default is the first element.
        max_channels: number of channels to plot.
        max_frames: number of frames for 2D-t plot.
        tag: tag of the plotted image on TensorBoard.
    """
    d = data[index].detach().cpu().numpy(
    ) if torch.is_tensor(data) else data[index]

    # H, W
    if d.ndim == 2:
        d = mtimage.image_transform(d)
        dataformats = "HW"
        writer.add_image(f"{tag}", d,
                         step, dataformats=dataformats)
        return

    # C, H, W
    if d.ndim == 3:
        d = [mtimage.image_transform(dd) for dd in d]
        if do_coloring:
            # coloring each channel to one canvas
            d = np.array(d, np.float32)[..., np.newaxis]
            d = colorization(d.transpose(3, 1, 2, 0))
            dataformats = "HWC"
            writer.add_image(f"{tag}", d[0],
                             step, dataformats=dataformats)
            return
        else:
            # plot each channel separately
            dataformats = 'HW'
            for idx, dd in enumerate(d):
                writer.add_image(f"{tag}_channel_{idx}", dd,
                                 step, dataformats=dataformats)
            return

    # C, H, W, D
    if d.ndim >= 4:
        d = [mtimage.image_transform(dd) for dd in d]
        if d[0].shape[-1] > max_frames:
            # sampling D to max_frames
            interval = int(d[0].shape[-1]/max_frames)
            d = [dd[..., ::interval] for dd in d]
        if do_coloring:
            # coloring each channel to one canvas
            d = np.array(d, np.float32).transpose(3, 1, 2, 0)
            d = colorization(d)
            d = (np.array(d, np.float32)*255).astype(np.uint8)
            writer._get_file_writer().add_summary(
                _gifims_to_summary(tag, d, colorspace=3),
                step,
            )
            return
        else:
            # plot each channel separately
            d = (np.array(d, np.float32)*255).astype(np.uint8)
            for idx, dd in enumerate(d):
                writer._get_file_writer().add_summary(
                    _gifims_to_summary(f"{tag}_channel_{idx}",
                                       dd.transpose(2, 0, 1), colorspace=1),
                    step,
                )
            return
