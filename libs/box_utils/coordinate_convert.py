# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np



def forward_convert(coordinate, with_label=True):
    """
    :param coordinate: format [x_c, y_c, w, h, theta]
    :return: format [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    boxes = []
    if with_label:
        for rect in coordinate:
            box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
            box = np.reshape(box, [-1, ])
            boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7], rect[5]])
    else:
        for rect in coordinate:
            box = cv2.boxPoints(((rect[0], rect[1]), (rect[2], rect[3]), rect[4]))
            box = np.reshape(box, [-1, ])
            boxes.append([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]])

    return np.array(boxes, dtype=np.float32)


def back_forward_convert(coordinate, with_label=True):
    """
    :param coordinate: format [x1, y1, x2, y2, x3, y3, x4, y4, (label)] 
    :param with_label: default True
    :return: format [x_c, y_c, w, h, theta, (label)]
    """

    boxes = []
    if with_label:
        for rect in coordinate:
            box = np.int0(rect[:-1])
            box = box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
            boxes.append([x, y, w, h, theta, rect[-1]])

    else:
        for rect in coordinate:
            box = np.int0(rect)
            box = box.reshape([4, 2])
            rect1 = cv2.minAreaRect(box)

            x, y, w, h, theta = rect1[0][0], rect1[0][1], rect1[1][0], rect1[1][1], rect1[2]
            boxes.append([x, y, w, h, theta])

    return np.array(boxes, dtype=np.float32)


def my_convert(coordinate, with_label=True):
    '''
    :param coordinate: format [x1, y1, x2, y2, x3, y3, x4, y4, (label)]
    :param with_label:
    :return: [x1, y1, x2, y2, h, label]  see as R2CNN
    '''
    boxes = []
    if with_label:
        for box in coordinate:
            rect = np.array(box[:-1])
            x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
            x3, y3 = rect[4], rect[5]
            h = np.sqrt((x2 - x3) * (x2 - x3) + (y2 - y3) * (y2 - y3))
            boxes.append([x1, y1, x2, y2, h, box[-1]])

    else:
        for box in coordinate:
            rect = np.array(box)
            x1, y1, x2, y2 = rect[0], rect[1], rect[2], rect[3]
            x3, y3 = rect[4], rect[5]
            h = np.sqrt((x2 - x3) * (x2 - x3) + (y2 - y3) * (y2 - y3))
            boxes.append([x1, y1, x2, y2, h])

    return np.array(boxes, dtype=np.float32)


def my_getnms_area(coordinate):
    '''
    get nms inclined box area
    :param coordinate: [x1, y1, x2, y2, h]
    :param with_label:
    :return: [xc, yc, w, h, theta]
    '''
    boxes = []
    for rect in coordinate:
        box = np.array(rect)
        x1, y1, x2, y2, h = box[0], box[1], box[2], box[3], box[4]
        if abs(x1 - x2) < 1e-6:
            x3 = x2 + h
            y3 = y2
        else:
            k1 = (y2 - y1) / (x2 - x1)
            if abs(k1) < 0.01:   # tan(pi/180)
                x3 = x2
                y3 = y2 + h
            else:
                kk = -1.0 / k1
                bias = np.sqrt(h * h / (1 + kk * kk))
                bias = bias if kk > 0 else -bias
                x3 = x2 + bias
                y3 = y2 + kk * (x3 - x2)

        x4 = x1 + x3 - x2
        y4 = y1 + y3 - y2

        #print([x1,y1,x2,y2,x3,y3,x4,y4])

        tmp = np.int0([x1, y1, x2, y2, x3, y3, x4, y4]).reshape([4, 2])
        rect2 = cv2.minAreaRect(tmp)
        x, y, w, h, theta = rect2[0][0], rect2[0][1], rect2[1][0], rect2[1][1], rect2[2]
        #print(x, y, w, h, theta)
        boxes.append([x, y, w, h, theta])

    return np.array(boxes, dtype=np.float32)


if __name__ == '__main__':

    coord1 = np.array([[392.1219, 152.3644, 456.0470, 153.3651, 42.0590],
                       [392.6227, 152.6220, 456.2122, 153.8851, 41.8631]])


    boxes = my_getnms_area(coord1)




