import logging
import os
import re
import shutil
import sys
import traceback
from copy import deepcopy
from tempfile import TemporaryDirectory
from uuid import uuid4

import cv2
import nibabel as nib
import numpy as np
import pydicom
import SimpleITK as sitk
import skimage.draw
import vtk
from jellyfish import levenshtein_distance
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
from scipy import interpolate
from vtkmodules.util.numpy_support import vtk_to_numpy

try:
    from .utils import stdout_redirected
except:
    stdout_redirected = None

zh_pattern = re.compile(r'[^x00-x7F]')
SOPClassUID_mapping = {'CT': ['1.2.840.10008.5.1.4.1.1.2', '1.2.840.10008.5.1.4.1.1.2.1'],
                       'CBCT': ['1.2.840.10008.5.1.4.1.1.2', '1.2.840.10008.5.1.4.1.1.2.1'],
                       'MR': ['1.2.840.10008.5.1.4.1.1.4', '1.2.840.10008.5.1.4.1.1.4.1'],
                       'STRUCT': '1.2.840.10008.5.1.4.1.1.481.3',
                       'PLAN': ['1.2.840.10008.5.1.4.1.1.481.5', '1.2.246.352.70.1.70'],
                       'PET':['1.2.840.10008.5.1.4.1.1.128', '1.2.840.10008.5.1.4.1.1.129'],
                       'DOSE': '1.2.840.10008.5.1.4.1.1.481.2'}

def smooth_contour(contour, smooth_coeff=0.1, fraction_of_points_to_keep=1.0):
    s = len(contour)*smooth_coeff
    tck, u = interpolate.splprep([contour[:, 0], contour[:, 1]], s=s)
    unew = np.arange(u[0], u[-1]+1e-6, 1.0/fraction_of_points_to_keep/len(u))
    out = interpolate.splev(unew, tck)
    return np.transpose(out)


def list_dcm_files(directory):
    '''
        find all dicom files in a directory

        :param directory: a directory contain dicom files, this directory
            can contain sub-directories
        :return: a list, which each item contrain dcm's SOPInstanceUID, and value contain
            path, SOPClassUID
    '''
    dcm_files = []
    for root, dirs, fns in os.walk(directory):
        for fn in fns:
            dcm_fn = os.path.join(root, fn)
            dcm = pydicom.read_file(dcm_fn, force=True)
            if hasattr(dcm, 'SOPClassUID'):
                if not hasattr(dcm, 'SOPInstanceUID'):
                    logging.warning(
                        '{} is a dicom file, but dose not have SOPInstanceUID'.format(dcm_fn))
                    continue
                dcm_files.append({
                    'path': dcm_fn,
                    'SOPClassUID': dcm.SOPClassUID,
                    'SOPInstanceUID': dcm.SOPInstanceUID,
                    'file': dcm,
                })
            else:
                logging.debug('{} is not a dicom file'.format(dcm_fn))
    return dcm_files


def get_modality_files(dcm_files, SOPClassUID):
    '''
        find all files of a special modality in dcm_files
    '''
    if not isinstance(SOPClassUID, list):
        SOPClassUID = [SOPClassUID]
    files_of_modality = []
    for item in dcm_files:
        if item['SOPClassUID'] in SOPClassUID:
            files_of_modality.append(item)
    return files_of_modality


def get_dicom_image_files(dcm_files, modalities=['CT', 'MR', 'PET']):
    '''
        find all dicom image files in dcm_files
    '''
    if not isinstance(modalities, list):
        modalities = [modalities]
    SOPClassUID_image = []
    for mod in modalities:
        SOPClassUID_image.extend(SOPClassUID_mapping[mod.upper()])
    return get_modality_files(dcm_files, SOPClassUID_image)


def get_dicom_structure_files(dcm_files):
    '''
        find all dicom structure files in dcm_files
    '''
    return get_modality_files(dcm_files, SOPClassUID_mapping['STRUCT'])


def get_dicom_dose_files(dcm_files):
    '''
        find all dose files in dcm_files
    '''
    return get_modality_files(dcm_files, SOPClassUID_mapping['DOSE'])


def get_dicom_plan_files(dcm_files):
    '''
        find all plan files in dcm_files
    '''
    return get_modality_files(dcm_files, SOPClassUID_mapping['PLAN'])


def get_frame_of_reference_UID(dcm):
    '''
        get FrameOfReferenceUID of a dicom file
    '''
    if hasattr(dcm, 'FrameOfReferenceUID'):
        return dcm.FrameOfReferenceUID
    else:
        if hasattr(dcm, 'ReferencedFrameOfReferenceSequence'):
            ReferencedFrameOfReferenceSequence = dcm.ReferencedFrameOfReferenceSequence
            if len(ReferencedFrameOfReferenceSequence) > 1:
                logging.warn(
                    'More than one ReferencedFrameOfReference found')
            for ReferencedFrameOfReference in ReferencedFrameOfReferenceSequence:
                if hasattr(ReferencedFrameOfReference, 'FrameOfReferenceUID'):
                    return ReferencedFrameOfReference.FrameOfReferenceUID


def get_structure_related_image_files(structure_dcm, dcm_files):
    '''
        find all relevant dicom images of a structure,
        1. if structure file have ContourImageSequence in RTReferencedSeriesSequence, use ReferencedSOPInstanceUID in structure
        2. else, if structure file have FrameOfReferenceUID in ReferencedFrameOfReference, use FrameOfReferenceUID
    '''
    find_image = False
    related_image_files = []
    if hasattr(structure_dcm, 'ReferencedFrameOfReferenceSequence'):
        ReferencedFrameOfReferenceSequence = structure_dcm.ReferencedFrameOfReferenceSequence
        if len(ReferencedFrameOfReferenceSequence) > 1:
            logging.warn('More than one ReferencedFrameOfReference found')
        if len(ReferencedFrameOfReferenceSequence) == 0:
            logging.error(
                'Cannot find ReferencedFrameOfReference in structure')
        for ReferencedFrameOfReference in ReferencedFrameOfReferenceSequence:
            if find_image:
                break
            if hasattr(ReferencedFrameOfReference, 'RTReferencedStudySequence'):
                try:
                    RTReferencedStudy = ReferencedFrameOfReference.RTReferencedStudySequence[0]
                    RTReferencedSeriesSequence = RTReferencedStudy.RTReferencedSeriesSequence
                    RTRefenecedSeries = RTReferencedSeriesSequence[0]
                    for ContourImage in RTRefenecedSeries.ContourImageSequence:
                        ReferencedSOPInstanceUID = ContourImage.ReferencedSOPInstanceUID
                        for item in dcm_files:
                            if item['SOPInstanceUID'] == ReferencedSOPInstanceUID:
                                find_image = True
                                related_image_files.append(item)
                                break
                        if not find_image:
                            logging.warn('Cannot find {} which referenced by structure'.format(
                                ReferencedSOPInstanceUID))
                except:
                    pass

        if not find_image:
            for ReferencedFrameOfReference in ReferencedFrameOfReferenceSequence:
                FrameOfReferenceUID = ReferencedFrameOfReference.FrameOfReferenceUID
                try:
                    for item in dcm_files:
                        if item['file'].FrameOfReferenceUID == FrameOfReferenceUID:
                            related_image_files.append(item)
                except:
                    pass

    return related_image_files


def get_dose_related_plan_file(dose_dcm, dcm_files):
    if hasattr(dose_dcm, 'ReferencedRTPlanSequence'):
        ReferencedRTPlanSequence = dose_dcm.ReferencedRTPlanSequence
        if len(ReferencedRTPlanSequence) > 1:
            logging.warn('More than one ReferencedRTPlanSequence found')
        if len(ReferencedRTPlanSequence) == 0:
            logging.error(
                'Cannot find ReferencedRTPlanSequence in rtdose')
        for ReferencedRTPlan in ReferencedRTPlanSequence:
            for item in dcm_files:
                if item['file'].SOPInstanceUID == ReferencedRTPlan.ReferencedSOPInstanceUID:
                    return item
    return None


def get_dose_related_structure_file(dose_dcm, dcm_files):
    if hasattr(dose_dcm, 'ReferencedStructureSetSequence'):
        ReferencedStructureSetSequence = dose_dcm.ReferencedStructureSetSequence
        if len(ReferencedStructureSetSequence) > 1:
            logging.warn('More than one ReferencedStructureSetSequence found')
        if len(ReferencedStructureSetSequence) == 0:
            logging.error(
                'Cannot find ReferencedStructureSetSequence in rtdose')
        for ReferencedStructureSet in ReferencedStructureSetSequence:
            for item in dcm_files:
                if item['file'].SOPInstanceUID == ReferencedStructureSet.ReferencedSOPInstanceUID:
                    return item
    return None


def get_plan_related_structure_file(plan_dcm, dcm_files):
    if hasattr(plan_dcm, 'ReferencedStructureSetSequence'):
        ReferencedStructureSetSequence = plan_dcm.ReferencedStructureSetSequence
        if len(ReferencedStructureSetSequence) > 1:
            logging.warn('More than one ReferencedStructureSetSequence found')
        if len(ReferencedStructureSetSequence) == 0:
            logging.error(
                'Cannot find ReferencedStructureSetSequence in rtdose')
        for ReferencedStructureSet in ReferencedStructureSetSequence:
            for item in dcm_files:
                if item['file'].SOPInstanceUID == ReferencedStructureSet.ReferencedSOPInstanceUID:
                    return item
    return None


def check_if_nonuniform_slice_thickness(dicom_image_files):
    """
    检查相邻两层的层厚是否一致
    """
    dicom_image_files_sorted = sorted(dicom_image_files, key=lambda t: float(
        t['file'].ImagePositionPatient[2]))
    if len(dicom_image_files_sorted) > 1:
        previous_z = dicom_image_files_sorted[0]['file'].ImagePositionPatient[2]
        gap_first = dicom_image_files_sorted[1]['file'].ImagePositionPatient[2] - previous_z
        for i in range(1, len(dicom_image_files_sorted)):
            current_z = dicom_image_files_sorted[i]['file'].ImagePositionPatient[2]
            gap = current_z - previous_z
            if np.abs(gap - gap_first) > 1e-6:
                return False
            previous_z = current_z
    return True


def find_one_rs_file(patient_dir, do_check=True):
    '''
    当前病人路径下必须有且只有一个RTStructure文件才会正确返回，否则返回None
    '''
    dcm_files = list_dcm_files(patient_dir)
    rs_files = get_dicom_structure_files(dcm_files)
    if len(rs_files) != 1:
        for num, info in enumerate([os.path.basename(i['path']) for i in rs_files]):
            if re.match(r'RTSTRUCT_2', info):
                return rs_files[num]['file'], rs_files[num]['path']
        return None, None

    if do_check and not is_rtdcm_valid(rs_files[0]['file'], rs_files[0]['path']):
        logging.error(
            'rt file: {} is not valid'.format(rs_files[0]['path']))
        return None, None

    return rs_files[0]['file'], rs_files[0]['path']


def read_uniform_image_sitk(dicom_image_files, sort=True):
    def execute_reader(img_its):
        reader = sitk.ImageSeriesReader()
        with TemporaryDirectory() as tmp:
            redirected_paths = []
            for t in img_its:
                if zh_pattern.search(t['path']):
                    new_path = os.path.join(tmp, str(uuid4()))
                    shutil.copy(t['path'], new_path)
                    redirected_paths.append(new_path)
                else:
                    redirected_paths.append(t['path'])
            reader.SetFileNames(redirected_paths)
            reader.SetLoadPrivateTags(True)
            image_sitk = reader.Execute()
            return image_sitk
    if sort:
        img_its = sorted(dicom_image_files, key=lambda t: float(
            t['file'].ImagePositionPatient[2]))
        if get_direction_from_dcm(dicom_image_files[0]['file'])[-1] < 0:
            img_its = list(reversed(img_its))
    else:
        img_its = dicom_image_files
    if stdout_redirected is not None:
        with stdout_redirected(stdout=sys.stderr):
            return execute_reader(img_its)
    else:
        return execute_reader(img_its)


def read_nonuniform_image_sitk(dicom_image_files, sort=True):
    if sort:
        img_its = sorted(dicom_image_files, key=lambda t: float(
            t['file'].ImagePositionPatient[2]))
        if get_direction_from_dcm(dicom_image_files[0]['file'])[-1] < 0:
            img_its = list(reversed(img_its))
    else:
        img_its = dicom_image_files
    image_sitk_list = []
    uniform_dicom_image_files = []
    previous_z = 0
    previous_gap = 0
    for idx, t in enumerate(img_its):
        if idx == 0:
            uniform_dicom_image_files.append(t)
            previous_z = float(t['file'].ImagePositionPatient[2])
            continue
        current_z = float(t['file'].ImagePositionPatient[2])
        gap = np.abs(current_z - previous_z)
        previous_z = current_z
        if idx == 1:
            previous_gap = gap
            uniform_dicom_image_files.append(t)
            continue
        if not np.allclose(gap, previous_gap, atol=1e-4):
            uniform_image_sitk = read_uniform_image_sitk(
                uniform_dicom_image_files)
            image_sitk_list.append(uniform_image_sitk)
            uniform_dicom_image_files = [t]
            previous_gap = gap
        else:
            uniform_dicom_image_files.append(t)
    uniform_image_sitk = read_uniform_image_sitk(uniform_dicom_image_files)
    image_sitk_list.append(uniform_image_sitk)

    return image_sitk_list


def sitk_get_image(patient_dir, ignore_nonuniform=True, modalities=['CT', 'MR', 'PET']):
    try:
        dicom_image_files = get_dicom_image_files(list_dcm_files(patient_dir),modalities)
        if ignore_nonuniform:
            return read_uniform_image_sitk(dicom_image_files)
        else:
            return read_nonuniform_image_sitk(dicom_image_files)
    except:
        logging.error(
            'sitk get image from [{}] failed'.format(patient_dir))
        return None


def is_rtdcm_valid(rt_dcm, rt_path):
    # StructureSetROISequence, ROIContourSequence, RTROIObservationsSequence must have same length
    try:
        ssrs_len = len(rt_dcm.StructureSetROISequence)
        rcs_len = len(rt_dcm.ROIContourSequence)
        rros_len = len(rt_dcm.RTROIObservationsSequence)
        if ssrs_len != rcs_len or ssrs_len != rros_len or rcs_len != rros_len:
            logging.error(
                'rt file {} is not valid because of unequal ssrs, rcs and rros length'.format(rt_path))
            return False
        ROINumber_set = []
        for StructureSetROI in rt_dcm.StructureSetROISequence:
            ROINumber_set.append(StructureSetROI.ROINumber)
        if len(set(ROINumber_set)) != len(ROINumber_set):
            logging.error(
                'rt file {} is not valid because of duplicated ROINumber'.format(rt_path))
            return False
        return True
    except:
        logging.error(
            'rt file {} is not valid because of exception'.format(rt_path))
        return False


def try_save_dcm(dcm, fn):
    try:
        pydicom.dcmwrite(fn, dcm)
    except:
        logging.error('file: {} is cannot be write'.format(
            os.path.basename(fn)))


def get_enc_from_rt_dcm(rt_dcm):
    try:
        enc = pydicom.charset.convert_encodings(rt_dcm.SpecificCharacterSet)[0]
    except:
        logging.debug('rt_dcm has no attribute SpecificCharacterSet')
        enc = 'utf-8'
    return enc


def get_roi_mapping_dict(rt_dcm, roi_names=None):
    roi_mapping = {}

    enc = get_enc_from_rt_dcm(rt_dcm)
    for ss_roi in rt_dcm.StructureSetROISequence:
        roi_index = int(ss_roi.ROINumber)
        roi_name = str(ss_roi.ROIName.encode(enc).decode()).lower()
        if roi_index in roi_mapping.keys():
            logging.error('{} and {} have same ROINumber {}'.format(
                roi_mapping[roi_index], roi_name, roi_index))
            return None
        if roi_names is not None:
            if roi_name not in roi_names:
                logging.debug('Skip roi [{}]'.format(roi_name))
                continue
            else:
                logging.debug('got roi name: [{}]'.format(roi_name))
        roi_mapping[roi_index] = roi_name

    if roi_names is not None:
        for roi_name in roi_names:
            if roi_name not in roi_mapping.values():
                # logging.warning('Cannot find {}'.format(roi_name))
                pass

    return roi_mapping


def rtstruct2stencilV1(image_sitk, rt_dcm, roi_names, fill_contour=True, inds_for_smooth=[]):
    roi_names_lower = [roi_name.lower() for roi_name in roi_names]
    if not len(roi_names_lower) == len(set(roi_names_lower)):
        logging.error(
            'rtstruct2stencil failed because there are duplicated roi names')
        return None

    roi_mapping = get_roi_mapping_dict(rt_dcm, roi_names=roi_names_lower)
    if roi_mapping is None:
        logging.error('rtstruct2stencil failed because roi mapping is None')
        return None

    W, H, Z = image_sitk.GetSize()
    label = np.zeros((Z, H, W, len(roi_names_lower)), dtype=np.bool)

    for roi_contour in rt_dcm.ROIContourSequence:
        ref_roi_number = int(roi_contour.ReferencedROINumber)
        if ref_roi_number not in roi_mapping:
            continue

        contours_each_slice = [(i, []) for i in range(Z)]

        if hasattr(roi_contour, 'ContourSequence'):
            # parse roi contour sequence
            for contour in roi_contour.ContourSequence:
                try:
                    contour_data = contour.ContourData
                except:
                    logging.debug('ContourData not exists')
                    continue
                if len(contour_data) % 3 != 0:
                    logging.debug('ContourData is corrupted')
                    continue
                contour_data = np.array(contour_data).reshape((-1, 3))
                contour_data = [image_sitk.TransformPhysicalPointToContinuousIndex(
                    t) for t in contour_data]
                contour_data = np.array(contour_data)
                if roi_names_lower.index(roi_mapping[ref_roi_number]) in inds_for_smooth:
                    z_coord = contour_data[:, 2][0]
                    contour_data = smooth_contour(contour_data[:, :2])
                    z_coord = np.repeat(z_coord, len(contour_data))
                    contour_data = np.concatenate(
                        (contour_data, z_coord[..., np.newaxis]), axis=1)
                if len(contour_data) < 5:
                    logging.debug('less than 5 points in contour data')
                    continue
                slice_index = contour_data[0, 2]
                if slice_index < 0:
                    logging.debug(
                        'Slice index < 0, data should be normalized')
                    continue
                if not (slice_index - round(slice_index)) < 1e-3:
                    logging.debug(
                        'Slice index cannot find corrsponding image')
                    continue
                slice_index = int(round(slice_index))
                if int(slice_index) >= Z:
                    logging.debug('contour is out of image')
                    continue
                contours_each_slice[slice_index][1].append(
                    contour_data)

            # convert contour to label
            for slice_index, contours in contours_each_slice:
                # get inner poly
                contours = list(filter(lambda t: len(t) >= 4, contours))

                # find inner poly
                inner_polys = []
                try:
                    polys = [(cont_index, Polygon(cont))
                             for cont_index, cont in enumerate(contours)]
                    inner_polys = []
                    for poly_index, poly in polys:
                        for poly_index_j, poly_j in polys:
                            if poly_index_j != poly_index and poly_j.contains(poly):
                                inner_polys.append(poly_index)
                                break
                except:
                    # not shapely found
                    masks = []
                    if len(contours) == 0:
                        continue
                    for contour in contours:
                        mask = np.zeros((H, W), dtype=np.float32)
                        cor_xy = contour[:, :2].tolist()
                        cv2.fillPoly(mask, np.int32([cor_xy]), 1)
                        mask = mask.astype(np.uint8)
                        masks.append(mask)

                    for idx, mask in enumerate(masks):
                        for jdx in range(idx+1, len(masks)):
                            union_area = (masks[jdx] | mask).sum()
                            if union_area == masks[jdx].sum():
                                inner_polys.append(idx)
                            elif union_area == mask.sum():
                                inner_polys.append(jdx)
                inner_polys = list(set(inner_polys))

                # draw mask
                if fill_contour:
                    func = skimage.draw.polygon
                else:
                    func = skimage.draw.polygon_perimeter

                for contour in contours:
                    rr, cc = func(contour[:, 1], contour[:, 0])
                    label[slice_index, rr, cc,
                          roi_names_lower.index(roi_mapping[ref_roi_number])] = 1
                for cont_index in inner_polys:
                    rr, cc = func(contours[cont_index]
                                  [:, 1], contours[cont_index][:, 0])
                    label[slice_index, rr, cc,
                          roi_names_lower.index(roi_mapping[ref_roi_number])] = 0

        else:
            logging.debug('{} not exists in ContourSequence'.format(
                roi_mapping[ref_roi_number]))

    return label.astype(np.bool)


def image_from_vtk_image(vtk_image):
    """Convert a vtk.vtkImageData to an itk.Image."""
    point_data = vtk_image.GetPointData()
    array = vtk_to_numpy(point_data.GetScalars())
    array = array.reshape(-1)
    is_vector = point_data.GetScalars().GetNumberOfComponents() != 1
    dims = list(vtk_image.GetDimensions())
    if is_vector and dims[-1] == 1:
        # 2D
        dims = dims[:2]
        dims.reverse()
        dims.append(point_data.GetScalars().GetNumberOfComponents())
    else:
        dims.reverse()
    array.shape = tuple(dims)
    # image = itk.image_view_from_array(array, is_vector)
    image = sitk.GetImageFromArray(array)

    dim = image.GetDimension()
    spacing = [1.0] * dim
    spacing[:dim] = vtk_image.GetSpacing()[:dim]
    image.SetSpacing(spacing)
    origin = [0.0] * dim
    origin[:dim] = vtk_image.GetOrigin()[:dim]
    image.SetOrigin(origin)
    # Todo: Add Direction with VTK 9
    return image


def rtstruct2stencil(image_sitk, rt_dcm, roi_names, old_direction=None, fill_contour=True, vis_debug=False):
    '''
        convert structures to numpy data array,  as vtk does not support direction,
        input image_sitk MUST have uniform direction
    :param: image_sitk: simpleItk Image
    :param: rt_dcm: dicom structure
    :param: roi_names: roi names to be extracted
    '''
    # check direction
    if not np.allclose(image_sitk.GetDirection(), [1, 0, 0, 0, 1, 0, 0, 0, 1], atol=1e-4):
        raise NotImplementedError('Non-uniform direction is not supported')

    roi_names_lower = [roi_name.lower() for roi_name in roi_names]
    if not len(roi_names_lower) == len(set(roi_names_lower)):
        logging.error(
            'rtstruct2stencil failed because there are duplicated roi names')
        return None

    roi_mapping = get_roi_mapping_dict(rt_dcm, roi_names=roi_names_lower)
    if roi_mapping is None:
        logging.error('rtstruct2stencil failed because roi mapping is None')
        return None

    W, H, Z = image_sitk.GetSize()
    label = np.zeros((Z, H, W, len(roi_names_lower)), dtype=np.bool)

    if old_direction is not None:
        direction = np.array(old_direction, np.float64)
        direction = direction.reshape(3, 3)
    else:
        direction = None
        logging.debug(
            'old direction is None, may cause errors when deal with oblique image')

    for roi_contour in rt_dcm.ROIContourSequence:
        ref_roi_number = int(roi_contour.ReferencedROINumber)
        if ref_roi_number not in roi_mapping:
            continue

        if hasattr(roi_contour, 'ContourSequence'):
            contours = vtk.vtkPolyData()
            points = vtk.vtkPoints()
            cells = vtk.vtkCellArray()

            # parse roi contour sequence
            for contour in roi_contour.ContourSequence:
                try:
                    contour_data = contour.ContourData
                except:
                    logging.debug('ContourData not exists')
                    continue
                if len(contour_data) % 3 != 0:
                    logging.debug('ContourData is corrupted')
                    continue

                numPoints = len(contour_data)//3
                polyLine = vtk.vtkPolyLine()
                for i in range(numPoints):
                    point = contour_data[3*i:3*(i+1)]
                    if direction is not None:
                        point = np.dot(point, direction)
                    pointID = points.InsertNextPoint(point)
                    polyLine.GetPointIds().InsertNextId(pointID)
                first_point = contour_data[:3]
                if direction is not None:
                    first_point = np.dot(first_point, direction)
                pointID = points.InsertNextPoint(first_point)
                polyLine.GetPointIds().InsertNextId(pointID)
                cells.InsertNextCell(polyLine)

            contours.SetPoints(points)
            contours.SetLines(cells)
            contours.Modified()

            cleanPolyData = vtk.vtkCleanPolyData()
            cleanPolyData.SetInputData(contours)
            cleanPolyData.Update()

            normalFilter = vtk.vtkPolyDataNormals()
            normalFilter.SetInputConnection(cleanPolyData.GetOutputPort())
            normalFilter.ConsistencyOn()
            triangle = vtk.vtkTriangleFilter()
            triangle.SetInputConnection(normalFilter.GetOutputPort())

            stripper = vtk.vtkStripper()
            stripper.SetInputConnection(triangle.GetOutputPort())
            stripper.Update()

            # Visualize vtkPolyData
            if vis_debug:
                renderer = vtk.vtkRenderer()
                renderWindow = vtk.vtkRenderWindow()
                renderWindow.SetWindowName('Polygon')
                renderWindow.AddRenderer(renderer)
                renderWindowInteractor = vtk.vtkRenderWindowInteractor()
                renderWindowInteractor.SetRenderWindow(renderWindow)

                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(contours)
                colors = vtk.vtkNamedColors()
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(colors.GetColor3d('white'))

                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(stripper.GetOutput())
                colors = vtk.vtkNamedColors()
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetColor(colors.GetColor3d('black'))

                renderer.AddActor(actor)
                renderer.SetBackground(colors.GetColor3d('ivory_black'))
                renderWindow.Render()
                renderWindowInteractor.Start()

            origin = image_sitk.GetOrigin()
            size = image_sitk.GetSize()
            spacing = image_sitk.GetSpacing()
            extent = 0, size[0]-1, 0, size[1]-1, 0, size[2]-1

            whiteImage = vtk.vtkImageData()
            whiteImage.SetSpacing(spacing)
            whiteImage.SetOrigin(origin)
            whiteImage.SetExtent(extent)
            whiteImage.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
            whiteImage.GetPointData().GetScalars().Fill(1)

            pol2stenc = vtk.vtkPolyDataToImageStencil()
            pol2stenc.SetInputConnection(stripper.GetOutputPort())
            pol2stenc.SetInformationInput(whiteImage)
            pol2stenc.Update()

            imgStenc = vtk.vtkImageStencil()
            imgStenc.SetInputData(whiteImage)
            imgStenc.SetStencilConnection(pol2stenc.GetOutputPort())
            imgStenc.ReverseStencilOff()
            imgStenc.SetBackgroundValue(0)
            imgStenc.Update()

            labelImage = image_from_vtk_image(imgStenc.GetOutput())
            roiArray = sitk.GetArrayFromImage(labelImage).astype(np.bool)
            label[..., roi_names_lower.index(
                roi_mapping[ref_roi_number])] += roiArray

        else:
            logging.debug('{} not exists in ContourSequence'.format(
                roi_mapping[ref_roi_number]))

    return label.astype(np.bool)


def get_roi_names_from_rt_dcm(rt_dcm):
    roi_names = set()
    enc = get_enc_from_rt_dcm(rt_dcm)
    for StructureSetROI in rt_dcm.StructureSetROISequence:
        roi_names.add(StructureSetROI.ROIName.encode(enc).decode())
    return list(roi_names)


def nearest_names(standard_names, name):
    dists = [levenshtein_distance(name, t) for t in standard_names]
    inds = np.argsort(dists)
    return list(np.array(standard_names)[inds])


def get_ROINumbers_from_rt_dcm(rt_dcm):
    ROINumbers = []
    for StructureSetROI in rt_dcm.StructureSetROISequence:
        ROINumbers.append(StructureSetROI.ROINumber)
    return ROINumbers


def fix_ROINumber_of_rt_dcm(rt_dcm):
    ROINumber_old_to_new = {}
    roi_id = 1
    enc = get_enc_from_rt_dcm(rt_dcm)
    for StructureSetROI in sorted(rt_dcm.StructureSetROISequence, key=lambda t: t.ROIName.encode(enc).decode()):
        ROINumber_old_to_new[StructureSetROI.ROINumber] = str(roi_id)
        roi_id += 1
    StructureSetROISequence = Sequence()
    for StructureSetROI in rt_dcm.StructureSetROISequence:
        # Fisrt: fix StructureSetROI
        StructureSetROI.ROINumber = ROINumber_old_to_new[StructureSetROI.ROINumber]
        StructureSetROISequence.append(StructureSetROI)
    ROIContourSequence = Sequence()
    RTROIObservationsSequence = Sequence()
    for ROIContour in rt_dcm.ROIContourSequence:
        # Second: fix ROIContour
        old_RefROINumber = ROIContour.ReferencedROINumber
        if old_RefROINumber not in ROINumber_old_to_new.keys():
            continue
        ROIContour.ReferencedROINumber = ROINumber_old_to_new[old_RefROINumber]
        ROIContourSequence.append(ROIContour)

        # Third: fix RTROIObservations
        RTROIObservations = Dataset()
        RTROIObservations.ObservationNumber = ROINumber_old_to_new[old_RefROINumber]
        RTROIObservations.ReferencedROINumber = ROINumber_old_to_new[old_RefROINumber]
        RTROIObservations.RTROIInterpretedType = ''
        RTROIObservations.ROIInterpreter = ''
        RTROIObservationsSequence.append(RTROIObservations)

    rt_dcm.StructureSetROISequence = StructureSetROISequence
    rt_dcm.ROIContourSequence = ROIContourSequence
    rt_dcm.RTROIObservationsSequence = RTROIObservationsSequence

    return rt_dcm


def get_roi_number(dcm, roi_name):
    roi_numbers = []
    enc = get_enc_from_rt_dcm(dcm)
    for StructureSetROI in dcm.StructureSetROISequence:
        if StructureSetROI.ROIName.encode(enc).decode().lower() == roi_name.lower():
            roi_numbers.append(StructureSetROI.ROINumber)
    return roi_numbers


def get_roi_center(dcm, roi_number):
    contours = []
    for ROIContour in dcm.ROIContourSequence:
        if ROIContour.ReferencedROINumber == roi_number:
            if hasattr(ROIContour, 'ContourSequence'):
                ContourSequence = ROIContour.ContourSequence
                for Contour in ContourSequence:
                    ContourData = Contour.ContourData
                    contours.extend(ContourData)
    contours = np.array(contours).reshape((-1, 3))
    center_x = np.mean(contours[:, 0])
    center_y = np.mean(contours[:, 1])
    return [center_x, center_y]


def write_npy_array(filename, array, eps=1e-5):
    """
        save an numpy array to file as small as possible
    """
    sum_uint8 = np.sum(array - array.astype(np.uint8))
    sum_int8 = np.sum(array - array.astype(np.int8))
    sum_uint16 = np.sum(array - array.astype(np.uint16))
    sum_int16 = np.sum(array - array.astype(np.int16))
    sum_float32 = np.sum(array - array.astype(np.float32))
    if abs(sum_uint8) < eps:
        np.save(filename, array.astype(np.uint8))
    elif abs(sum_int8) < eps:
        np.save(filename, array.astype(np.int8))
    elif abs(sum_uint16) < eps:
        np.save(filename, array.astype(np.uint16))
    elif abs(sum_int16) < eps:
        np.save(filename, array.astype(np.int16))
    elif abs(sum_float32) < eps:
        np.save(filename, array.astype(np.float32))
    else:
        np.save(filename, array)


def write_nifti(filename, array, eps=1e-5):
    """
        save an numpy array to nifti as small as possible
    """
    sum_uint8 = np.sum(array - array.astype(np.uint8))
    sum_int8 = np.sum(array - array.astype(np.int8))
    sum_uint16 = np.sum(array - array.astype(np.uint16))
    sum_int16 = np.sum(array - array.astype(np.int16))
    sum_float32 = np.sum(array - array.astype(np.float32))
    if abs(sum_uint8) < eps:
        img_nib = nib.Nifti1Image(array.astype(np.uint8), np.eye(4))
    elif abs(sum_int8) < eps:
        img_nib = nib.Nifti1Image(array.astype(np.int8), np.eye(4))
    elif abs(sum_uint16) < eps:
        img_nib = nib.Nifti1Image(array.astype(np.uint16), np.eye(4))
    elif abs(sum_int16) < eps:
        img_nib = nib.Nifti1Image(array.astype(np.int16), np.eye(4))
    elif abs(sum_float32) < eps:
        img_nib = nib.Nifti1Image(array.astype(np.float32), np.eye(4))
    nib.save(img_nib, filename)


def make_dir_if_not_exist(directory, action='ask'):
    if os.path.exists(directory):
        if action == 'remove':
            logging.warning(
                '{} exists, do remove and create'.format(directory))
            shutil.rmtree(directory)
        elif action == 'ask':
            logging.error(
                '{} exists, enter Y to remove and create, enter G to ignore and go on'.format(directory))
            key = input()
            if key == 'Y':
                shutil.rmtree(directory)
            elif key == 'G':
                pass
            else:
                logging.warning('cancel')
                return False
        elif action == 'ignore':
            logging.warning('{} exists, continue'.format(directory))
            return True
        elif action == 'abort':
            logging.warning('{} exists, abort'.format(directory))
            return False
        else:
            raise ValueError('unsupported action {}'.format(action))

    if not os.path.exists(directory):
        os.mkdir(directory)

    return True


def add_empty_roi(dcm, name):
    existing_ROINumbers = get_ROINumbers_from_rt_dcm(dcm)

    def create_unique_ROINumber(existing_ROINumbers):
        current_ROINumber = 1
        while True:
            if current_ROINumber in existing_ROINumbers:
                current_ROINumber += 1
                continue
            return current_ROINumber

    current_ROINumber = str(create_unique_ROINumber(existing_ROINumbers))
    StructureSetROISequence = deepcopy(dcm.StructureSetROISequence)
    StructureSetROI = Dataset()
    try:
        StructureSetROI.ReferencedFrameOfReferenceUID = dcm.FrameOfReferenceUID
    except:
        StructureSetROI.ReferencedFrameOfReferenceUID = \
            dcm.ReferencedFrameOfReferenceSequence[0].FrameOfReferenceUID
    StructureSetROI.ROINumber = current_ROINumber
    StructureSetROI.ROIName = name
    StructureSetROI.ROIVolume = '0.0'
    StructureSetROI.ROIGenerationAlgorithm = 'MANUAL'
    StructureSetROISequence.append(StructureSetROI)
    ROIContourSequence = deepcopy(dcm.ROIContourSequence)
    ROIContour = Dataset()
    ROIContour.ROIDisplayColor = '255\\0\\0'
    ROIContour.ReferencedROINumber = current_ROINumber
    ROIContour.ContourSequence = Sequence()
    ROIContourSequence.append(ROIContour)

    RTROIObservationsSequence = deepcopy(dcm.RTROIObservationsSequence)
    RTROIObservations = Dataset()
    RTROIObservations.ObservationNumber = current_ROINumber
    RTROIObservations.ReferencedROINumber = current_ROINumber
    RTROIObservations.RTROIInterpretedType = 'ORGAN'
    RTROIObservations.ROIInterpreter = ''
    RTROIObservationsSequence.append(RTROIObservations)

    dcm.StructureSetROISequence = StructureSetROISequence
    dcm.ROIContourSequence = ROIContourSequence
    dcm.RTROIObservationsSequence = RTROIObservationsSequence
    return dcm


def read_npy(filename):
    return np.load(filename)


def read_nifti(filename):
    return nib.load(filename).get_fdata()


def resolve_loader(loader):
    '''
        将loader字符串转化为对象
    '''
    if loader == 'numpy':
        prefix = 'npy'
        writer = write_npy_array
        reader = read_npy
    elif loader == 'nibabel':
        prefix = 'nii.gz'
        writer = write_nifti
        reader = read_nifti
    else:
        raise ValueError('unsupported image loader {}'.format(loader))

    return prefix, reader, writer


def resolve_dtype(dtype):
    if dtype == 'float32':
        data_type = np.float32
    elif dtype == 'float64':
        data_type = np.float64
    elif dtype == 'int32':
        data_type = np.int32
    elif dtype == 'uint32':
        data_type = np.uint32
    elif dtype == 'int16':
        data_type = np.int16
    elif dtype == 'uint16':
        data_type = np.uint16
    elif dtype == 'int8':
        data_type = np.int8
    elif dtype == 'uint8':
        data_type = np.uint8
    else:
        raise NotImplementedError('unsupported dtype {}'.format(dtype))

    return data_type


def resolve_attr(obj, attr):
    if hasattr(obj, attr):
        return getattr(obj, attr)
    else:
        return None


def check_diff(data, dtype, dpath, eps=1e-2):
    diff = np.abs(np.sum(data - data.astype(dtype)))
    if diff > eps:
        logging.warning(
            'casting dtype of {} caused loss of data accuracy '.format(dpath))


def get_direction_from_dcm(dcm):
    direction_xy = dcm.ImageOrientationPatient
    direction_xy = np.array(direction_xy).reshape(2, 3)
    direction_z = np.cross(direction_xy[0], direction_xy[1])
    direction = np.concatenate((direction_xy, direction_z[np.newaxis, ...]))
    return direction.reshape(-1).tolist()


def if_direction_is_oblique(direction):
    rad_xyz = cv2.Rodrigues(np.array(direction).reshape(3, 3))[0].squeeze()
    for rad in rad_xyz:
        if np.abs(rad) < 1e-6 or np.abs(np.abs(rad) - np.pi) < 1e-6:
            continue
        else:
            return True
    return False


def which_rois_are_empty(rt_dcm, roi_names):
    roi_names_lower = [roi_name.lower() for roi_name in roi_names]
    roi_mapping = get_roi_mapping_dict(rt_dcm, roi_names=roi_names_lower)

    result = []
    for roi_contour in rt_dcm.ROIContourSequence:
        ref_roi_number = int(roi_contour.ReferencedROINumber)
        if ref_roi_number not in roi_mapping:
            continue
        if not hasattr(roi_contour, 'ContourSequence'):
            roi_name = roi_mapping[ref_roi_number]
            target = roi_name
            for raw_roi_name in roi_names:
                if roi_name == raw_roi_name.lower():
                    target = raw_roi_name
                    break
            result.append(target)

    return result
