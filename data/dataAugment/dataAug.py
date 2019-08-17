# data augmentation for Oriented Bounding Boxes
# date: 2019-07-26
# reference: https://github.com/maozezhong/CV_ToolBox

# include augmentation operations:
# 1. add gaussian noise
# 2. change light
# 3. rotate img
# 4. crop img
# 5. shift img
# 6. flip img

import random
import cv2
import os
import math
import numpy as np
from skimage.util import random_noise
from skimage import exposure
from data.dataAugment.xml_helper import *
import copy
import scipy.io as sio


class DataAugmentForOBB():
    def __init__(self, rotation_rate=0.5, max_rotation_angle=5,
                 crop_rate=0.5, shift_rate=0.5, change_light_rate=0.5,
                 add_noise_rate=0.5, flip_rate=0.5):
        self.rotation_rate = rotation_rate
        self.max_rotation_angle = max_rotation_angle
        self.crop_rate = crop_rate
        self.shift_rate = shift_rate
        self.change_light_rate = change_light_rate
        self.add_noise_rate = add_noise_rate
        self.flip_rate = flip_rate

    def _addNoise(self, img):
        '''
        :param img: img array
        :return: img array with noise
        '''
        return random_noise(img, mode='gaussian', clip=True) * 255

    def _changeLight(self, img):
        # random.seed(int(tmie.time()))
        # flag>1 darker, flag<1 brighter
        flag = random.uniform(0.5, 1.5)
        return exposure.adjust_gamma(img, flag)

    def _rotateImg(self, img, bboxes, angle=5, scale=1.):
        '''
        :param img: img array (h,w,c)
        :param bboxes: [[x1,y1,x2,y2,x3,y3,x4,y4],...]
        :param angle: rotate angles
        :param scale:
        :return: rot_img, rot_boxes
        '''
        # ---------------------rot img------------------
        h = img.shape[0]
        w = img.shape[1]
        # angle in radians
        rangle = np.deg2rad(angle)
        # new rot image width and height
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # affine transform
        rot_img = cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)
        # print('rot_mat shape is:', rot_mat.shape) (2, 3)

        # -------------------rot boxes------------------
        rot_boxes = list()
        for bbox in bboxes:
            # rot_mat is the rotate matrix
            p1 = np.dot(rot_mat, np.array([bbox[0], bbox[1], 1]))
            p2 = np.dot(rot_mat, np.array([bbox[2], bbox[3], 1]))
            p3 = np.dot(rot_mat, np.array([bbox[4], bbox[5], 1]))
            p4 = np.dot(rot_mat, np.array([bbox[6], bbox[7], 1]))
            rboxes = np.array([p1,p2,p3,p4]).flatten()  # [rx1,ry1,rx2,ry2,rx3,ry3,rx4,ry4]
            rot_boxes.append(rboxes)
        # print('rot_boxes shape is:', rot_boxes[0].shape)
        return rot_img, rot_boxes

    def _cropImg(self, img, bboxes):
        # the cropped img should contain all gt bboxes
        # ----------------crop image-------------
        h = img.shape[0]
        w = img.shape[1]
        # the mininum crop box contains all gt
        x_min = w
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[2]
            y2 = bbox[3]
            x3 = bbox[4]
            y3 = bbox[5]
            x4 = bbox[6]
            y4 = bbox[7]
            arrx = np.array([x1, x2, x3, x4])
            arry = np.array([y1, y2, y3, y4])
            x_min = min(x_min, arrx.min())
            x_max = max(x_max, arrx.max())
            y_min = min(y_min, arry.min())
            y_max = max(y_max, arry.max())

        # distance to the img border
        d_to_left = x_min
        d_to_right = w - x_max
        d_to_top = y_min
        d_to_bottom = h - y_max

        # random scale the minimum box
        crop_x_min = int(x_min - random.uniform(d_to_left/3, d_to_left))  # default 0
        crop_y_min = int(y_min - random.uniform(d_to_top/3, d_to_top))
        crop_x_max = int(x_max + random.uniform(d_to_right/4, d_to_right))
        crop_y_max = int(y_max + random.uniform(d_to_right/4, d_to_bottom))

        # ensure don't beyond the border
        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(w, crop_x_max)
        crop_y_max = min(h, crop_y_max)

        crop_img = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        # ---------------crop bbox------------------
        crop_bboxes = list()
        for bbox in bboxes:
            x1 = bbox[0] - crop_x_min
            y1 = bbox[1] - crop_y_min
            x2 = bbox[2] - crop_x_min
            y2 = bbox[3] - crop_y_min
            x3 = bbox[4] - crop_x_min
            y3 = bbox[5] - crop_y_min
            x4 = bbox[6] - crop_x_min
            y4 = bbox[7] - crop_y_min
            crop_bboxes.append([x1, y1, x2, y2, x3, y3, x4, y4])

        return crop_img, crop_bboxes

    def _shiftImg(self, img, bboxes):
        # the shifted img should contain all gt boxes
        h = img.shape[0]
        w = img.shape[1]
        # the mininum crop box contains all gt
        x_min = w
        x_max = 0
        y_min = h
        y_max = 0
        for bbox in bboxes:
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[2]
            y2 = bbox[3]
            x3 = bbox[4]
            y3 = bbox[5]
            x4 = bbox[6]
            y4 = bbox[7]
            arrx = np.array([x1, x2, x3, x4])
            arry = np.array([y1, y2, y3, y4])
            x_min = min(x_min, arrx.min())
            x_max = max(x_max, arrx.max())
            y_min = min(y_min, arry.min())
            y_max = max(y_max, arry.max())

        # distance to the img border
        d_to_left = x_min
        d_to_right = w - x_max
        d_to_top = y_min
        d_to_bottom = h - y_max

        # the shift pixels
        sx = random.uniform(-(d_to_left - 1) / 3, (d_to_right - 1) / 3)
        sy = random.uniform(-(d_to_top - 1) / 3, (d_to_bottom - 1) / 3)

        M = np.float32([[1, 0, sx], [0, 1, sy]])
        shift_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

        # --------------------shift bboxes-------------------
        shift_bboxes = list()
        for bbox in bboxes:
            x1 = bbox[0] + sx
            y1 = bbox[1] + sy
            x2 = bbox[2] + sx
            y2 = bbox[3] + sy
            x3 = bbox[4] + sx
            y3 = bbox[5] + sy
            x4 = bbox[6] + sx
            y4 = bbox[7] + sy
            shift_bboxes.append([x1, y1, x2, y2, x3, y3, x4, y4])

        return shift_img, shift_bboxes

    def _flipImg(self, img, bboxes):

        flip_img = copy.deepcopy(img)
        # if random.random() < 0.5:
        #     horizon = True  # horizon flip
        #     flip_img = cv2.flip(flip_img, 1)
        # else:
        #     horizon = False  # vertical flip
        #     flip_img = cv2.flip(flip_img, 0)
        horizon = True
        flip_img = cv2.flip(flip_img, 1)

        # -----------------flip bboxes-------------
        h = img.shape[0]
        w = img.shape[1]
        flip_bboxes = list()
        for bbox in bboxes:
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[2]
            y2 = bbox[3]
            x3 = bbox[4]
            y3 = bbox[5]
            x4 = bbox[6]
            y4 = bbox[7]
            if horizon:
                tmp_box = [w - x1, y1, w - x2, y2, w - x3, y3, w - x4, y4]
                flip_bboxes.append(tmp_box)
            else:
                tmp_box = [x1, h - y1, x2, h - y2, x3, h - y3, x4, h - y4]
                flip_bboxes.append(tmp_box)

        return flip_img, flip_bboxes

    def dataAugment(self, img, bboxes, op):
        '''
        :param img: previous img array
        :param bboxes: previous bboxes [-1, 8]
        :return: augment img, augment bboxes
        '''
        if op.startswith('crop'):
            aimg, abboxes = self._cropImg(img, bboxes)
        elif op.startswith('shift'):
            aimg, abboxes = self._shiftImg(img, bboxes)
        elif op == 'noise':
            aimg = self._addNoise(img)
            abboxes = bboxes
        elif op.startswith('rotate'):
            angle = random.randint(-20, 20)
            #scale = random.uniform(0.8, 1.0)
            aimg, abboxes = self._rotateImg(img, bboxes, angle)
        elif op == 'light':
            aimg = self._changeLight(img)
            abboxes = bboxes
        elif op == 'flip':
            aimg, abboxes = self._flipImg(img, bboxes)
        else:
            assert False, "Unrecognised augment operation:" + op

        return aimg, abboxes


def draw_img(save_dir, img_name, img, bboxes=None):
    '''
    draw img with bounding boxes
    :param img: img array
    :param bboxes: bboxes list [N,8]
    :return:
    '''
    # resize the image and its gt label coordinates
    n = len(bboxes)
    gtbox = np.zeros((n, 6))
    h, w = img.shape[:2]
    if h < w:
        new_h = 600
        new_w = 600 * w // h
    else:
        new_h = 600 * h // w
        new_w = 600

    re_img = cv2.resize(img, (new_w, new_h))  # [fxsrc.cols, fysrc.rows]

    for i, rect in enumerate(bboxes):
        x1, y1, x2, y2 = rect[0]*new_w//w, rect[1]*new_h//h, rect[2]*new_w//w, rect[3]*new_h//h
        x3, y3, x4, y4 = rect[4]*new_w//w, rect[5]*new_h//h, rect[6]*new_w//w, rect[7]*new_h//h
        theta = 0
        if abs(x1 - x2) < 1e-6:
            theta = 0
        else:
            k = - (y2 - y1) / (x2 - x1)
            theta = np.degrees(np.arctan(k))

        up_xc = (x1 + x2) / 2
        up_yc = (y1 + y2) / 2
        theta = int(theta)
        # draw degree of each detected bone
        cv2.putText(re_img, text=str(theta), org=(int(up_xc), int(up_yc + 20)), fontFace=1, fontScale=1, thickness=2,
                    color=(0, 0, 255))

        gtbox[i, :] = [x1, y1, x2, y2, theta, up_yc]

        box = np.int0([x1,y1,x2,y2,x3,y3,x4,y4])
        box = box.reshape([4, 2])
        rect1 = cv2.minAreaRect(box)  # x,y,w,h,theta
        rect1 = cv2.boxPoints(rect1)
        rect1 = np.int0(rect1)
        cv2.drawContours(re_img, [rect1], -1, (0, 0, 255), 1)

    save_path = os.path.join(save_dir, 'output',  img_name)
    cv2.putText(re_img,
                text=str(n),
                org=((re_img.shape[1]) // 2, (re_img.shape[0]) // 2),
                fontFace=3,
                fontScale=2,
                color=(0, 255, 255))
    cv2.imwrite(save_path, re_img)

    gtbox = gtbox[gtbox[:, 5].argsort()]
    gtpath = os.path.join(save_dir, 'gtbox', img_name[:-4]+'.mat')
    sio.savemat(gtpath, {'gtbox':gtbox})




if __name__ == '__main__':

    trans = ['crop', 'shift', 'rotate1', 'rotate2']   # augment 9 images per img


    dataAug = DataAugmentForOBB()
    # VOC dataset
    src_img_path = r'D:\validset\JPEGImages'
    src_xml_path = r'D:\validset\Annotation'

    # save draw_img with bboxes
    save_dir = r'D:\validset'
    aug_dir = r'E:\3AllRBox\VOCdevkit\RandJPEGImages'
    out_xml_path = r'E:\3AllRBox\VOCdevkit\RandAnnotation'

    for file in os.listdir(src_img_path):

        img_path = os.path.join(src_img_path, file)
        xml_path = os.path.join(src_xml_path, file[:-4] + '.xml')
        coords = read_xml(xml_path)  # [-1, 8]

        # attention: draw_img cache affect the clean img so destroy the following for (draw_img)
        img = cv2.imread(img_path)
        draw_img(save_dir, file, img, coords)  # draw raw img with bboxes

        # for op in trans:
        #     img = cv2.imread(img_path)
        #     aug_img, aug_bboxes = dataAug.dataAugment(img, coords, op)
        #     draw_img(aug_dir, op+'-'+file, aug_img, [])  # save augment img with bboxes
        #     generate_xml(op+'-'+file, aug_bboxes, aug_img.shape, out_xml_path)
            # print('------------------augment ', file, ' finished------------------')
