#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Python 
@File    ：medical_img_resample.py
@IDE     ：PyCharm 
@Author  ：Andy
@Date    ：2022/12/5 16:03 
@Maibox  : 13266082905@163.com
'''

import numpy as np
import SimpleITK as sitk

def resample_img_spacing(image, outimageFilepath, new_spacing=[1.0, 1.0, 1.0], is_label=True):
    '''
      image:
      outimageFilepath:
      new_spacing: [n,n,n]
      new_spacing: x,y,z
      is_label: if True, using Interpolator `sitk.sitkNearestNeighbor`
    '''
    try:
        size = np.array(image.GetSize())    # 读取原图尺寸
        print("size:{}".format(size))
        spacing = np.array(image.GetSpacing())  # 读取原图spacing
        print("spacing:{}".format(spacing))
        new_spacing = np.array(new_spacing)
        new_size = size * spacing / new_spacing     # 计算新尺寸
        new_spacing_refine = size * spacing / new_size  # 计算新spacing
        new_spacing_refine = [float(s) for s in new_spacing_refine]
        new_size = [int(s) for s in new_size]
        print("new_size:{}".format(new_size))
        print("new_space:{}".format(new_spacing))

        resample = sitk.ResampleImageFilter()
        resample.SetOutputDirection(image.GetDirection())
        resample.SetOutputOrigin(image.GetOrigin())
        resample.SetSize(new_size)
        resample.SetOutputSpacing(new_spacing_refine)
        if is_label:
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample.SetInterpolator(sitk.sitkLinear)
        newimage = resample.Execute(image)
        sitk.WriteImage(newimage, outimageFilepath)
    except:
        print('该数据有问题，未重采样')
        sitk.WriteImage(image, outimageFilepath)

def resize_mask_itk(ori_img, target_img, resamplemethod=sitk.sitkNearestNeighbor):
    """
    用itk方法将原始图像resample到与目标图像一致
    :param ori_img: 原始需要对齐的itk图像
    :param target_img: 要对齐的目标itk图像
    :param resamplemethod: itk插值方法: sitk.sitkLinear-线性  sitk.sitkNearestNeighbor-最近邻
    :return:img_res_itk: 重采样好的itk图像
    使用示范：
    import SimpleITK as sitk
    target_img = sitk.ReadImage(target_img_file)
    ori_img = sitk.ReadImage(ori_img_file)
    img_r = resize_image_itk(ori_img, target_img, resamplemethod=sitk.sitkLinear)
    """
    target_Size = target_img.GetSize()  # 目标图像大小  [x,y,z]
    target_Spacing = target_img.GetSpacing()  # 目标的体素块尺寸    [x,y,z]
    target_origin = target_img.GetOrigin()  # 目标的起点 [x,y,z]
    target_direction = target_img.GetDirection()  # 目标的方向 [冠,矢,横]=[z,y,x]

    # itk的方法进行resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ori_img)  # 需要重新采样的目标图像
    # 设置目标图像的信息
    resampler.SetSize(target_Size)  # 目标图像大小
    resampler.SetOutputOrigin(target_origin)
    resampler.SetOutputDirection(target_direction)
    resampler.SetOutputSpacing(target_Spacing)
    # 根据需要重采样图像的情况设置不同的dype
    if resamplemethod == sitk.sitkNearestNeighbor:
        resampler.SetOutputPixelType(sitk.sitkUInt8)  # 近邻插值用于mask的，保存uint8
    else:
        resampler.SetOutputPixelType(sitk.sitkFloat32)  # 线性插值用于PET/CT/MRI之类的，保存float32
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    itk_img_resampled = resampler.Execute(ori_img)  # 得到重新采样后的图像
    return itk_img_resampled


if __name__ == "__main__":
    # Image resample
    nii_path = r"D:\xx.nii.gz"
    in_save_path = r"D:\xx_in.nii.gz"
    de_save_path = r"D:\xx_de.nii.gz"
    img = sitk.ReadImage(nii_path)
    old_spacing = np.array(img.GetSpacing())
    # Increase resolution
    resample_img_spacing(img, in_save_path, new_spacing=[old_spacing[0]*0.9, old_spacing[1]*0.9, old_spacing[2]*0.9])
    # Descend resolution
    resample_img_spacing(img, de_save_path, new_spacing=[old_spacing[0]*1.1, old_spacing[1]*1.1, old_spacing[2]*1.1])

    # Mask resample
    mask_nii_path = r"D:\xx_mask.nii.gz"
    mask_save_path = r"D:\xx_mask_resample.nii.gz"
    mask = sitk.ReadImage(mask_nii_path)
    taget_img = sitk.ReadImage(in_save_path)
    mask_resample = resize_mask_itk(mask, taget_img, resamplemethod=sitk.sitkNearestNeighbor)
    sitk.WriteImage(mask_resample, mask_save_path)
