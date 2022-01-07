from copy import deepcopy
import logging

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import skimage.morphology as skmo
from nibabel.affines import voxel_sizes
from nibabel.orientations import (apply_orientation, io_orientation,
                                  ornt_transform)

# 判断是否需要分离3D Resample，用2D的方式Resample，因为体素各向异性过大时，
# 3D Resample会造成较多的伪影
ANISOTROPY_SPACING_RATIO_TO_SEPARATE_RESAMPLE = 3.0
SITK_INTERP_MODE = {0: sitk.sitkNearestNeighbor,
                    1: sitk.sitkLinear, 3: sitk.sitkBSpline}


def reorientation(img_sitk):
    '''
        将img_sitk的direction重新定位到(1., 0., 0., 0., 1., 0., 0., 0., 1.)

        img_sitk: 必须是LPS(SPL indeed)的坐标系，即sitk.GetArrayFromImage(img_sitk).shape必须是(D, H, W)的numpy shape
    '''
    spacing3d = img_sitk.GetSpacing()
    old_direction = img_sitk.GetDirection()
    new_direction = (1., 0., 0., 0., 1., 0., 0., 0., 1.)

    if np.allclose(old_direction, new_direction, atol=1e-6):
        return img_sitk

    I, J, K = img_sitk.GetSize()
    coords = []
    for i in [0, I-1]:
        for j in [0, J-1]:
            for k in [0, K-1]:
                coords.append([i, j, k])
    coords = [img_sitk.TransformIndexToPhysicalPoint(t) for t in coords]
    coords = np.array(coords)
    min_i, max_i = np.min(coords[:, 0]), np.max(coords[:, 0])
    min_j, max_j = np.min(coords[:, 1]), np.max(coords[:, 1])
    min_k, max_k = np.min(coords[:, 2]), np.max(coords[:, 2])
    new_origin = [min_i, min_j, min_k]

    new_size = [0, 0, 0]
    new_size[0] = int(round((max_i-min_i)/spacing3d[0]))+1
    new_size[1] = int(round((max_j-min_j)/spacing3d[1]))+1
    new_size[2] = int(round((max_k-min_k)/spacing3d[2]))+1

    resampleFilter = sitk.ResampleImageFilter()
    resampleFilter.SetOutputSpacing(spacing3d)
    resampleFilter.SetOutputOrigin(new_origin)
    resampleFilter.SetOutputDirection(new_direction)
    resampleFilter.SetSize(new_size)

    out_img_sitk = resampleFilter.Execute(img_sitk)

    return out_img_sitk


def reorientation_nib(nib_obj):
    '''
        将nib_obj的orientation改为LPS(SPL indeed)， 重新定位后origin为(0, 0, 0),
        direction为(1., 0., 0., 0., 1., 0., 0., 0., 1.)

        nib_obj: nib.Nifti1Image
    '''
    data, affine = nib_obj.get_fdata(), nib_obj.affine
    src_ornt = io_orientation(affine)
    spacing = voxel_sizes(affine)
    spacing = spacing[src_ornt[:, 0].astype(np.int64)]

    # we use axcodes: LPS (SPL indeed)
    dst_ornt = np.array([[2, 1],
                         [1, -1],
                         [0, -1]])
    transform = ornt_transform(src_ornt, dst_ornt)
    data = apply_orientation(data, transform)
    img_sitk = sitk.GetImageFromArray(data)
    img_sitk.SetSpacing(spacing)
    affine = make_affine_from_sitk(img_sitk)
    nib_obj_transformed = nib.Nifti1Image(data, affine)

    return nib_obj_transformed


def align_orientation(img_sitk, ref_img_sitk):
    '''
        将img_sitk重新定位到ref_img_sitk的direction，img_sitk的spacing必须与ref_img_sitk的spacing相同

        img_sitk: 需要align的sitk图像
        ref_slices: 参考的sitk图像，用于获取目标的origin和direction
    '''
    assert ref_img_sitk.GetSpacing() == img_sitk.GetSpacing()

    resampleFilter = sitk.ResampleImageFilter()
    resampleFilter.SetOutputSpacing(ref_img_sitk.GetSpacing())
    resampleFilter.SetOutputOrigin(ref_img_sitk.GetOrigin())
    resampleFilter.SetOutputDirection(ref_img_sitk.GetDirection())
    resampleFilter.SetSize(ref_img_sitk.GetSize())

    img_sitk_aligned = resampleFilter.Execute(img_sitk)

    return img_sitk_aligned

# TODO: need to rewrite this function
# def align_orientation_nib(array, ref_nib_obj):
#     '''
#         将array重新定位到ref_nib_obj的orientation，再用ref_nib_obj.affine来初始化nib_obj并输出，
#         array的spacing必须与ref_nib_obj的spacing相同
#     '''
#     src_ornt = np.array([[2, 1],
#                          [1, -1],
#                          [0, -1]])
#     dst_ornt = io_orientation(ref_nib_obj.affine)
#     transform = ornt_transform(src_ornt, dst_ornt)
#     array_transformed = apply_orientation(array, transform)
#     nib_obj_transformed = nib.Nifti1Image(
#         array_transformed, ref_nib_obj.affine)
#     return nib_obj_transformed


def make_affine_from_sitk(img_sitk):
    '''
        计算nib.Nifti1Image的affine矩阵，img_sitk必须是LPS(SPL indeed)的坐标系
    '''
    if img_sitk.GetDepth() <= 0:
        return np.eye(4)

    rot = [img_sitk.TransformContinuousIndexToPhysicalPoint(p)
           for p in ((0, 0, 1),
                     (0, 1, 0),
                     (1, 0, 0),
                     (0, 0, 0))]
    rot = np.array(rot)
    affine = np.concatenate([
        np.concatenate([rot[0:3] - rot[3:], rot[3:]], axis=0),
        [[0.], [0.], [0.], [1.]]
    ], axis=1)
    affine = np.transpose(affine)
    affine = np.matmul(np.diag([-1., -1., 1., 1.]), affine)
    return affine


def image_transform(image, clip_value=None, clip_percent=None, standardization='MINMAX0', ret_stat=False):
    if clip_value is not None and clip_percent is not None:
        raise ValueError('only one limitation can be apply')
    if clip_value is not None:
        low = clip_value[0]
        high = clip_value[1]
    elif clip_percent is not None:
        background_value = image.min()
        low = np.percentile(image[image > background_value], clip_percent[0])
        high = np.percentile(image[image > background_value], clip_percent[1])
    else:
        low = None
        high = None
    if low is not None or high is not None:
        image = np.clip(image, low, high)
    if standardization == 'MINMAX0':
        shift = image.min() if low is None else low
        scale = (image.max() - image.min()) if high is None else high - low
        if scale == 0:
            image = image - shift
        else:
            image = (image - shift) / scale
    elif standardization == 'MINMAX1':
        shift = image.min() if low is None else low
        scale = (image.max() - image.min()) if high is None else high - low
        if scale == 0:
            image = image - shift
        else:
            image = (image - shift) / scale * 2 - 1
    elif standardization == 'Z-SCORE':
        shift = np.mean(image)
        scale = np.std(image)
        image = (image - shift) / scale
    elif standardization == 'NOT':
        shift = None
        scale = None
    else:
        raise ValueError(
            'not supported standardization {}'.format(standardization))

    if ret_stat:
        return image, shift, scale
    else:
        return image


def reverse_image_transform(image, shift, scale, standardization='MINMAX0'):
    if shift is not None and scale is not None:
        if scale == 0:
            logging.warning('scale is 0')
            return image
        if standardization == 'MINMAX1':
            return (image + 1) / 2 * scale + shift
        return image * scale + shift
    else:
        return image


def whether_to_separate_resample(spacing, anisotropy_threshold=ANISOTROPY_SPACING_RATIO_TO_SEPARATE_RESAMPLE):
    do_separate = (np.max(spacing) / np.min(spacing)) > anisotropy_threshold
    return do_separate


def resample_sitk(img_sitk, spacing, size, default_value=None, interpolator=sitk.sitkLinear):
    ref_image_itk = sitk.Image(size, img_sitk.GetPixelIDValue())
    ref_image_itk.SetOrigin(img_sitk.GetOrigin())
    ref_image_itk.SetDirection(img_sitk.GetDirection())
    ref_image_itk.SetSpacing(spacing)

    # 防止图像上下界出现空的图像
    new_img_sitk = sitk.ZeroFluxNeumannPad(
        img_sitk, [0, 0, 10], [0, 0, 10])
    if default_value is None:
        defaultPixelValue = 0
    else:
        defaultPixelValue = default_value

    new_img_sitk = sitk.Resample(
        new_img_sitk, ref_image_itk, interpolator=interpolator, defaultPixelValue=defaultPixelValue)
    return new_img_sitk


def resample_patient(array, old_new_spacing, new_size=None, interp_order=0, background_value=None):
    '''
        当old_spacing或new_spacing的各向异性ratio大于阈值时，会进行2D的resample再合并，若小于阈值，
        则直接进行3D的resample

        array: D, H, W numpy array
        old_new_spacing: [old_spacing, new_spacing] x, y, z order
        new_size: D, H, W shape
        interp_order: interpolation order
        background_value: 如果是数值，则在图像边缘进行constand pad，如果是None，则采用ZeroFluxNeumannPad，
                    对应scipy.zoom里的nearest模式
    '''
    assert len(array.shape) == 3
    assert array.dtype != np.bool, 'bool type array must be convert to uint8'
    img_sitk = sitk.GetImageFromArray(array)
    old_spacing, new_spacing = old_new_spacing
    old_size = array.shape[::-1]  # DHW -> WHD

    target_spacing = [new if new is not None else old for old,
                      new in zip(old_spacing, new_spacing)]
    if new_size is not None:
        target_size = new_size[::-1]
    else:
        target_size = [int(round(t0*t1/t2))
                       for t0, t1, t2 in zip(old_size, old_spacing, target_spacing)]
    target_size, old_size = list(target_size), list(old_size)
    if np.allclose(target_size, old_size, rtol=1e-4, atol=1e-4):
        return array

    img_sitk.SetSpacing(old_spacing)
    if whether_to_separate_resample(old_spacing):
        do_separate_resample = True
    elif whether_to_separate_resample(target_spacing):
        do_separate_resample = True
    else:
        do_separate_resample = False
    axis = 2  # 永远按照Z轴进行切分

    if do_separate_resample:
        # 首先按照axis一层一层进行resample
        target_size_along_axis = deepcopy(target_size)
        target_size_along_axis[axis] = old_size[axis]
        target_spacing_along_axis = deepcopy(target_spacing)
        target_spacing_along_axis[axis] = old_spacing[axis]
        img_sitk = resample_sitk(
            img_sitk, target_spacing_along_axis, target_size_along_axis, background_value, SITK_INTERP_MODE[interp_order])
        # 然后将再对axis进行resample, order为nearest
        img_sitk = resample_sitk(
            img_sitk, target_spacing, target_size, background_value, sitk.sitkNearestNeighbor)
    else:
        img_sitk = resample_sitk(
            img_sitk, target_spacing, target_size, background_value, SITK_INTERP_MODE[interp_order])

    return sitk.GetArrayFromImage(img_sitk)


def get_connected_component(array):
    assert array.dtype == np.bool
    lmap, num_cc = skmo.label(array, return_num=True)
    bincount = np.bincount(lmap.flat)[1:]
    return lmap, num_cc, bincount


def keep_largest_cc(array):
    array_keep = np.zeros(array.shape, np.bool)
    lmap, num_cc, bincount = get_connected_component(array)
    array_keep[lmap == (np.argmax(bincount)+1)] = True
    return array_keep


def get_bbox_3D(arr):
    a = np.any(arr, axis=(1, 2))
    b = np.any(arr, axis=(0, 2))
    c = np.any(arr, axis=(0, 1))

    amin, amax = np.where(a)[0][[0, -1]]
    bmin, bmax = np.where(b)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]

    return amin, amax, bmin, bmax, cmin, cmax
