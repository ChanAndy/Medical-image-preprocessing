# -*- coding: utf-8 -*-
"""
Created on Tuesday Januray 26 2021

@author: cky

Used for NII file format calculation of segmentation coefficient DSC
"""

import nibabel as nib
import numpy as np
import os

# The split ROI files drawn by different doctors should be in the nii_path folder
nii_path = r"D:..."
nii_list = os.listdir(nii_path)
nii_length = len(nii_list)
counter = 0
dscs = 0
for nii_name in nii_list:
    # print(nii_name)
    if nii_name.split("voi")[1].split(".")[0] == '2':
        print(nii_name)
        counter = counter + 1
        # Modify the code according to the actual naming of your files
        doctor_1 = nib.load(os.path.join(nii_path, nii_name))
        doctor_2 = nib.load(os.path.join(nii_path, nii_name.split("voi")[0]+'voi3.nii'))
        # Obtain a three-dimensional matrix
        data_1 = doctor_1.get_data()
        data_2 = doctor_2.get_data()
        # In my file, both NII files have the same image size, and I'm sure you do, too
        (x, y, z) = doctor_1.shape
        # In order to calculate the DSC of the 3D segmented image
        # the indexes of the 2D image need to be superimposed
        tps = 0
        fps = 0
        fns = 0
        dsc = 0
        # The DSC of three-dimensional ROI segmentation is calculated by superimposing two-dimensional indexes
        for i in range(0, z):
            slice_1 = data_1[:, :, i]
            slice_2 = data_2[:, :, i]
            point_true = np.sum(slice_1)
            point_result = np.sum(slice_2)
            tp = np.sum(slice_1 * slice_2)
            fp = point_result - tp
            fn = point_true - tp
            tps = tps + tp
            fps = fps + fp
            fns = fns + fn
        dsc = 2 * tps / (2 * tps + fps + fns)
        dscs = dscs + dsc
        print("nii_name:{} dsc:{}".format(nii_name.split("voi")[0], dsc))
average_dsc = dscs / counter
print("average_dsc:{}".format(average_dsc))

# Test the code for just one example
"""
doctor_1 = nib.load(r"....nii")
doctor_2 = nib.load(r"....nii")

data_1 = doctor_1.get_data()
data_2 = doctor_2.get_data()
(x, y, z) = doctor_1.shape

tps = 0
fps = 0
fns = 0

for i in range(0, z):    
    slice_1 = data_1[:, :, i]
    slice_2 = data_2[:, :, i]
    point_true = np.sum(slice_1)
    point_result = np.sum(slice_2)
    tp = np.sum(slice_1 * slice_2)
    fp = point_result - tp
    fn = point_true - tp
    tps = tps + tp
    fps = fps + fp
    fns = fns + fn
dsc_1 = 2 * tps / (2 * tps + fps + fns)
print("dsc_1:{}".format(dsc_1))
"""
