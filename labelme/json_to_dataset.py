'''
reference labelme_json_to_dataset
here aim to voc_to_mask.png for U-Net
'''

import numpy as np
import base64
import os
import os.path as osp

import PIL.Image
import scipy.io as sio
import skimage.io as io

from labelme import utils


def voc_to_mask(img_path, contours, png_dir, mask_dir):

    filename = osp.basename(img_path).replace('jpg', 'png')   # get the last filename of the path
    # print(filename)

    with open(img_path, 'rb') as f:
        imageData = f.read()
        imageData = base64.b64encode(imageData).decode('utf-8')
    img = utils.img_b64_to_arr(imageData)
    # print('img shape:', img.shape)
    # print(np.max(img))

    label_name_to_value = {'_background_': 0, 'bone': 1}

    lbl = utils.shapes_to_label(img.shape, contours, label_name_to_value)
    # print('mask shape:', lbl.shape)
    # print(np.max(lbl))

    PIL.Image.fromarray(img).save(osp.join(png_dir, filename))
    utils.lblsave(osp.join(mask_dir, filename), lbl)



if __name__ == '__main__':

    src_img_path = r'E:\Unet\test'
    src_mat_path = r'E:\ClearMats'
    png_dir = r'E:\Unet\pngImages'
    mask_dir = r'E:\Unet\labelImages'

    # a = io.imread(osp.join(src_img_path, 'L-4-4-001.jpg'), as_gray=True)
    # print('gray shape is:', a.shape)
    # print('gray max is:', np.max(a))
    # print('gray min is:', np.min(a))

    if not osp.exists(png_dir):
        os.mkdir(png_dir)

    if not osp.exists(mask_dir):
        os.mkdir(mask_dir)

    for file in os.listdir(src_img_path):
        img_path = osp.join(src_img_path, file)
        # with open(img_path, 'rb') as f:
        #     imageData = f.read()
        #     imageData = base64.b64encode(imageData).decode('utf-8')
        # img = utils.img_b64_to_arr(imageData)
        #
        # PIL.Image.fromarray(img).save(osp.join(r'E:\Unet\test\png', file.replace('jpg', 'png')))

        mat_path = osp.join(src_mat_path, file+'.mat')
        mat = sio.loadmat(mat_path)
        contours = []
        x = mat['x']
        y = mat['y']
        n = len(x) // 4 * 4
        for i in range(0, n, 4):
            xx = x[i:i+4]
            yy = y[i:i+4]
            bbox = zip(xx, yy)  # [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
            contours.append(bbox)

        voc_to_mask(img_path, contours, png_dir, mask_dir)
