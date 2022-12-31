
import numpy as np
import nibabel as nib
from scipy.ndimage import zoom
import h5py
import os
global nii_name
import pandas as pd
# import skimage
from skimage.transform import resize
import SimpleITK as sitk
import scipy.ndimage
# import cv2


def nii_loader(nii_path):
    print('#Loading ', nii_path, '...')
    data = nib.load(nii_path)

    return data

def my_resize(o_data, transform_size = None, transform_rate = None):
    print('#Resizing...')
    data = o_data
    print("--Original size:", data.shape)
    if transform_size:
        o_width, o_height, o_queue = data.shape
        width, height, queue = transform_size
        # zoom就是用插值的方法把原矩阵转化为目标大小
        data = zoom(data, (width/o_width, height/o_height, queue/o_queue))
    elif transform_rate:
        data = zoom(data, transform_rate)

    print("--Transofmed size:", data.shape)
    return data

def centre_window_cropping(o_data, reshapesize = None):
    print('#Centre window cropping...')
    data = o_data
    or_size = data.shape
    target_size = (reshapesize[0], reshapesize[1], or_size[2])

    # pad if or_size is smaller than target_size
    if (target_size[0] > or_size[0]) | (target_size[1] > or_size[1]):
        if target_size[0] > or_size[0]:
            pad_size = int((target_size[0] - or_size[0]) / 2)
            data = np.pad(data, ((pad_size, pad_size), (0, 0), (0, 0)))
        if target_size[1] > or_size[1]:
            pad_size = int((target_size[1] - or_size[1]) / 2)
            data = np.pad(data, ((0, 0), (pad_size, pad_size), (0, 0)))

    # centre_window_cropping
    cur_size = data.shape
    centre_x = float(cur_size[0] / 2)
    centre_y = float(cur_size[1] / 2)
    dx = float(target_size[0] / 2)
    dy = float(target_size[1] / 2)
    data = data[int(centre_x - dx + 1):int(centre_x + dx), int(centre_y - dy + 1): int(centre_y + dy), :]

    data = my_resize(data, transform_size=target_size)

    return data

def getListIndex(arr, value) :
    dim1_list = dim2_list = dim3_list = []
    if (arr.ndim == 3):
        index = np.argwhere(arr == value)
        dim1_list = index[:, 0].tolist()
        dim2_list = index[:, 1].tolist()
        dim3_list = index[:, 2].tolist()

    else:
        raise ValueError('The ndim of array must be 3!!')

    return dim1_list, dim2_list, dim3_list

def Max_min_normalizing(o_data):
    print('#Max_min_normalizing...')
    data = o_data
    minn = np.min(data)
    maxx = np.max(data)
    data = (data - minn) / (maxx - minn)

    return data

def equalize_hist(im, nbr_bins=256):
    """对一幅灰度图像进行直方图均衡化"""
    print("# Equalizing...")
    # 图像直方图统计
    imhist, bins = np.histogram(im.flatten(), nbr_bins)
    # 累积分布函数
    cdf = imhist.cumsum()
    cdf = 255.0 * cdf / cdf[-1]
    # 使用累积分布函数的线性插值，计算新的像素值
    im2 = np.interp(im.flatten(), bins[:-1], cdf)  # 分段线性插值函数
    return im2.reshape(im.shape), cdf


def ROI_cutting(o_data, o_roi, expend_voxel=0):
    print('#ROI cutting...')
    data = o_data
    roi = o_roi

    [I1, I2, I3] = getListIndex(roi, 1)
    d1_min = min(I1)
    d1_max = max(I1)
    d2_min = min(I2)
    d2_max = max(I2)
    d3_min = min(I3)
    d3_max = max(I3)
    print(d3_min, d3_max)

    if expend_voxel > 0:
        d1_min -= expend_voxel
        d1_max += expend_voxel
        d2_min -= expend_voxel
        d2_max += expend_voxel

        d1_min = d1_min if d1_min > 0 else 0
        d1_max = d1_max if d1_max < data.shape[0]-1 else data.shape[0]-1
        d2_min = d2_min if d2_min > 0 else 0
        d2_max = d2_max if d2_max < data.shape[1]-1 else data.shape[1]-1

    data = data[d1_min:d1_max, d2_min:d2_max, d3_min:d3_max]
    print(data.shape)
    roi = roi[d1_min:d1_max, d2_min:d2_max, d3_min:d3_max]

    print("--Cutting size:", data.shape)
    return data, roi


def make_h5_data(o_data, o_roi=None, label=None, h5_save_path=None,count = None,count1 = None):
    print('#Make h5 data...')
    data = o_data
    roi = o_roi
    if (h5_save_path):
        for i, divided_data in enumerate(data):
            if not os.path.exists(os.path.join(h5_save_path, str(count))):
                os.makedirs(os.path.join(h5_save_path, str(count)))
            save_file_name = os.path.join(h5_save_path, str(count), str(count) + '_' + str(i+1) + '.h5')
            with h5py.File(save_file_name, 'a') as f:
                print("--h5 file path:", save_file_name, '    -label:', label, '    -size:', divided_data.shape)
                f['Data'] = divided_data
                f['Label'] = [label]

def make_h5_data_new(o_data, o_mask, h5_save_path=None, count=None):
    print('#Make h5 data...')
    data = o_data
    mask = o_mask
    count_h5 = 0
    if (h5_save_path):
        for divided_data, divided_mask in zip(data, mask):
            if not os.path.exists(os.path.join(h5_save_path, str(count))):
                os.makedirs(os.path.join(h5_save_path, str(count)))
            save_file_name = os.path.join(h5_save_path, str(count), str(count) + '_' + str(count_h5+1) + '.h5')
            with h5py.File(save_file_name, 'a') as f:
                print("--h5 file path:", save_file_name, '    -size:', divided_data.shape)
                f['Data'] = divided_data
                f['Mask'] = divided_mask
            count_h5 += 1

def block_dividing(o_data, deep = None, step = None):
    print('#Block dividing...')
    data = o_data
    data_group = []
    o_data_deep = data.shape[2]

    if o_data_deep <= deep:
        tmp_data = np.zeros((data.shape[0], data.shape[1], deep))
        tmp_data[:, :, 0:o_data_deep] = data
        blocks = 1
        tmp_data = tmp_data
        data_group.append(tmp_data)

    else:
        blocks = (o_data_deep - deep) // step + 2
        if (o_data_deep - deep) % step == 0:
            blocks -= 1
        for i in range(blocks-1):
            tmp_data = data[:, :, (0 + i * step): (deep + i * step)]
            data_group.append(tmp_data)
        # tmp_data = np.zeros((data.shape[0],data.shape[1],deep))
        # tmp_data[:,:,0:(o_data_deep-(deep+i*step))] = data[:,:,(deep+i*step):o_data_deep]
        tmp_data = data[:, :, o_data_deep - deep:o_data_deep]
        data_group.append(tmp_data)

    print("--Block size:", tmp_data.shape,
          " Divided number:(%d)"%(blocks))

    return data_group, blocks

if __name__ == "__main__":

    """预设参数"""
    reshapesize = (384, 384)
    deep = 1
    step = 1
    # count = 0
    internal_validation_count = 0
    outernal_validation_count = 0

    """开始处理"""
    all_clients_path = "/media/root/3339482d-9d23-44ee-99a0-85e517217d15/CKY/Federated_learning_projection/Prostate_segmentation/Prostate_segmentation_data/All_clients_data"
    save_path = "/media/root/3339482d-9d23-44ee-99a0-85e517217d15/CKY/Federated_learning_projection/Prostate_segmentation/FedDG-ELCFS-main/preprocessing_dataset"
    client_list = os.listdir(all_clients_path)
    for client_id in client_list:
        # count = 0
        # 创建保存路径
        save_client_path = os.path.join(save_path, str(client_id))
        if not os.path.exists(save_client_path):
            os.makedirs(save_client_path)
        # 定位数据路径
        client_path = os.path.join(all_clients_path, client_id)
        img_path = os.path.join(client_path, "Image")
        img_list = os.listdir(img_path)
        print("img_list:{}".format(img_list))
        mask_path = os.path.join(client_path, "Mask")
        mask_list = os.listdir(mask_path)
        for data_id in img_list:
            # case num
            count = int(data_id.split(".")[0].split("_")[1])
            print("count:{}".format(count))
            # 载入数据
            data_metrix = nii_loader(os.path.join(img_path, data_id))
            mask_metrix = nii_loader(os.path.join(mask_path, data_id))
            # 转换数据格式
            mask_arr = np.array(mask_metrix.dataobj, dtype='float32')
            img_arr = np.array(data_metrix.dataobj, dtype='float32')
            mask_arr[mask_arr < 1] = 0
            mask_arr[mask_arr >= 1] = 1
            # 把ROI附近的矩形区域的原图提取出来
            img_arr, mask_arr = ROI_cutting(img_arr, mask_arr, expend_voxel=10)
            # 中值滤波
            # img_arr = scipy.ndimage.median_filter(img_arr, (4, 4, 4))
            # 直方图均衡化
            # img_arr, _ = equalize_hist(img_arr)
            # 把原图放到固定大小的黑图中
            img_arr = centre_window_cropping(img_arr, reshapesize=reshapesize)
            mask_arr = centre_window_cropping(mask_arr, reshapesize=reshapesize)
            # mask_arr[mask_arr < 1] = 0
            # mask_arr[mask_arr >= 1] = 1
            # 图像最大最小归一化
            img_arr = Max_min_normalizing(img_arr)
            # 组成固定大小的二维图像矩阵
            img_arr = block_dividing(img_arr, deep=deep, step=step)
            mask_arr = block_dividing(mask_arr, deep=deep, step=step)
            # 保存二维图像矩阵至h5格式
            # save_h5_path = os.path.join(save_client_path, )
            make_h5_data_new(img_arr[0], mask_arr[0], h5_save_path=save_client_path, count=count)

