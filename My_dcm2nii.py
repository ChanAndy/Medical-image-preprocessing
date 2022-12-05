"""
参考：https://blog.csdn.net/JianJuly/article/details/81214408
"""
import SimpleITK as sitk
import os
sep = os.sep

def dcm2nii(file_path, save_path):
    # Dicom序列所在文件夹路径（在我们的实验中，该文件夹下有多个dcm序列，混合在一起）
    # file_path = r'E:\@data_hcc_rna_mengqi\new\human_FCM\01\ADC'
    # 获取该文件下的所有序列ID，每个序列对应一个ID， 返回的series_IDs为一个列表
    series_IDs = sitk.ImageSeriesReader.GetGDCMSeriesIDs(file_path)
    # 查看该文件夹下的序列数量
    nb_series = len(series_IDs)

    # 通过ID获取该ID对应的序列所有切片的完整路径， series_IDs[1]代表的是第二个序列的ID
    # 如果不添加series_IDs[1]这个参数，则默认获取第一个序列的所有切片路径
    series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(file_path, series_IDs[0])

    # 新建一个ImageSeriesReader对象
    series_reader = sitk.ImageSeriesReader()

    # 通过之前获取到的序列的切片路径来读取该序列
    series_reader.SetFileNames(series_file_names)

    # 获取该序列对应的3D图像
    image3D = series_reader.Execute()

    # 将image转换为scan
    # 查看该3D图像的尺寸
    print(image3D.GetSize())
    sitk.WriteImage(image3D, os.path.join(save_path, 'nifti.nii.gz'))
    # sitk.WriteImage(image3D, save_path)
    print('save succed')

# 保存未成功转换的图像
import dicom2nifti
import os

patient_path = r"D:xxx\wide_field"  # dicom file path
patient_list = os.listdir(patient_path)
save_path = r"D:xxx\DICOM_to_NII"    # save path
if not os.path.exists(save_path):
    os.makedirs(save_path)

for patient_name in patient_list:
    print("patient_name:{}".format(patient_name))
    dicom_path = os.path.join(patient_path, patient_name)
    save_path_final = os.path.join(save_path, patient_name + '.nii.gz')
    dicom2nifti.dicom_series_to_nifti(dicom_path, save_path_final, reorient_nifti=False)



