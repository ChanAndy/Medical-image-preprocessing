#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Python 
@File    ：dicom_to_nii.py
@IDE     ：PyCharm 
@Author  ：Andy
@Date    ：2022/11/10 16:50 
@Maibox  : 13266082905@163.com
'''

"""
参考：https://blog.csdn.net/Along1617188/article/details/121233156
"""
import SimpleITK as sitk
import numpy as np
import os
import shutil

def format_conversion(work_path):
    '''
    :params work_path:  work dir path
                        ---root
                            --sud_dir(fixed dir)
                            --other_files
    :To Do:             get dicom dir
    '''
    case_list = os.listdir(path = work_path)
    case_list.sort()
    for case in case_list:
        # print('case = {}'.format(case))
        subdir = os.listdir(os.path.join(work_path, case))
        for ssubdir in subdir:
            if os.path.isdir(os.path.join(work_path, case, ssubdir)):
                # print('case = {}'.format(os.path.join(work_path, case, ssubdir)))
                image_path = os.path.join(work_path, case, ssubdir)
                read_write_dicom(image_path)

def read_write_dicom(dicom_dir_path):
    '''
    :params dicom_dir_path: dicom dir
    :To Do: read dicom and write to nii.gz
    '''
    reader = sitk.ImageSeriesReader()
    img_name = reader.GetGDCMSeriesFileNames(dicom_dir_path)
    reader.SetFileNames(img_name)
    image = reader.Execute()
    image_array = sitk.GetArrayFromImage(image) # z y x
    # print(image_array.shape)
    image_out = sitk.GetImageFromArray(image_array)
    image_out.SetOrigin(image.GetOrigin())  # recovery ori size
    image_out.SetSpacing(image.GetSpacing())
    image_out.SetDirection(image.GetDirection())
    sitk.WriteImage(image_out, 'mr_image.nii.gz')
    if os.path.isfile(os.path.join(os.path.abspath(os.path.dirname(dicom_dir_path)), 'mr_image.nii.gz')):
        os.remove(os.path.join(os.path.abspath(os.path.dirname(dicom_dir_path)), 'mr_image.nii.gz'))
    shutil.move(os.path.join(os.getcwd(), 'mr_image.nii.gz'), os.path.abspath(os.path.dirname(dicom_dir_path)))

if __name__ == '__main__':
    root_path = r'F:\project\xxx\Task\ROI_Testis_02'
    format_conversion(root_path)


# 保存未成功转换的图像
import dicom2nifti
import os

if __name__ == '__main__':
    patient_path = r"D:\NCI-ISBI 2013\NCI-ISBI 2013 DICOM Testing\manifest-7v55qffK2424752658836301389\Prostate-3T\Prostate3T-02-0005\04-27-2005-NA-MR prostaat kanker detectie WDSmc MCAPRODETW-04961\4.000000-t2tsetra-13941"
    patient_list = os.listdir(patient_path)
    save_path = r"D:\NCI-ISBI 2013\NCI-ISBI 2013 NII Testing\Prostate3T-02-0005.nii.gz"

    dicom2nifti.dicom_series_to_nifti(patient_path, save_path, reorient_nifti=False)



