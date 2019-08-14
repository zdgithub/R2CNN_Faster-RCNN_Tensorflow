# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import cv2
from libs.label_name_dict.label_dict import LABEl_NAME_MAP

from libs.configs import cfgs


def draw_box_with_color(img_batch, boxes, text):

    def draw_box_cv(img, boxes, text):
        img = img + np.array(cfgs.PIXEL_MEAN)
        boxes = boxes.astype(np.int64)
        img = np.array(img * 255 / np.max(img), np.uint8)
        for box in boxes:
            xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]

            color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            cv2.rectangle(img,
                          pt1=(xmin, ymin),
                          pt2=(xmax, ymax),
                          color=color,
                          thickness=2)

        text = str(text)
        cv2.putText(img,
                    text=text,
                    org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                    fontFace=3,
                    fontScale=1,
                    color=(255, 0, 0))

        # img = np.transpose(img, [2, 1, 0])
        img = img[:, :, ::-1]
        return img

    img_tensor = tf.squeeze(img_batch, 0)
    # color = tf.constant([0, 0, 255])
    img_tensor_with_boxes = tf.py_func(draw_box_cv,
                                       inp=[img_tensor, boxes, text],
                                       Tout=[tf.uint8])

    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))

    return img_tensor_with_boxes


def draw_box_with_color_rotate(img_batch, boxes, text):

    def draw_box_cv(img, boxes, text):
        img = img + np.array(cfgs.PIXEL_MEAN)
        boxes = boxes.astype(np.int64)
        img = np.array(img * 255 / np.max(img), np.uint8)
        for box in boxes:
            x_c, y_c, w, h, theta = box[0], box[1], box[2], box[3], box[4]
            rect = ((x_c, y_c), (w, h), theta)
            rect = cv2.boxPoints(rect)
            rect = np.int0(rect)
            color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            cv2.drawContours(img, [rect], -1, color, 3)

        text = str(text)
        cv2.putText(img,
                    text=text,
                    org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                    fontFace=3,
                    fontScale=1,
                    color=(255, 0, 0))

        img = img[:, :, ::-1]
        return img

    img_tensor = tf.squeeze(img_batch, 0)
    img_tensor_with_boxes = tf.py_func(draw_box_cv,
                                       inp=[img_tensor, boxes, text],
                                       Tout=[tf.uint8])

    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))

    return img_tensor_with_boxes


def draw_gtbox_my(img_batch, boxes, text):

    def draw_box_cv(img, boxes, text):
        img = img + np.array(cfgs.PIXEL_MEAN)
        img = np.array(img * 255 / np.max(img), np.uint8)

        for box in boxes:
            box = np.array(box)
            x1, y1, x2, y2, h = box[0], box[1], box[2], box[3], box[4]
            # get four points
            if abs(x1 - x2) < 1e-6:
                x3 = x2 + h
                y3 = y2
            else:
                k1 = (y2 - y1) / (x2 - x1)
                if abs(k1) < 0.01:  # tan(pi/180)
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

            tmp = np.int0([x1, y1, x2, y2, x3, y3, x4, y4]).reshape([4, 2])
            rect2 = cv2.minAreaRect(tmp)
            x_c, y_c = np.int0(rect2[0][0]), np.int0(rect2[0][1])

            rect = cv2.boxPoints(rect2)
            rect = np.int0(rect)

            color = (0, 0, 255)
            cv2.drawContours(img, [rect], -1, color, 3)
            cv2.line(img, (np.int0(x1), np.int0(y1)), (np.int0(x2), np.int0(y2)), (0, 255, 0), 3)
            cv2.circle(img, (np.int0(x1), np.int0(y1)), 1, (0, 255, 0), 8)

        text = str(text)
        cv2.putText(img,
                    text=text,
                    org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                    fontFace=3,
                    fontScale=1,
                    color=(255, 0, 0))

        img = img[:, :, ::-1]
        return img

    img_tensor = tf.squeeze(img_batch, 0)
    img_tensor_with_boxes = tf.py_func(draw_box_cv,
                                       inp=[img_tensor, boxes, text],
                                       Tout=[tf.uint8])

    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))

    return img_tensor_with_boxes



def draw_boxes_with_categories(img_batch, boxes, scores):

    def draw_box_cv(img, boxes, scores):
        img = img + np.array(cfgs.PIXEL_MEAN)
        boxes = boxes.astype(np.int64)
        img = np.array(img*255/np.max(img), np.uint8)

        num_of_object = 0
        for i, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]

            score = scores[i]

            num_of_object += 1
            color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            cv2.rectangle(img,
                          pt1=(xmin, ymin),
                          pt2=(xmax, ymax),
                          color=color,
                          thickness=2)
            cv2.rectangle(img,
                          pt1=(xmin, ymin),
                          pt2=(xmin+120, ymin+15),
                          color=color,
                          thickness=-1)
            cv2.putText(img,
                        text=str(score),
                        org=(xmin, ymin+10),
                        fontFace=1,
                        fontScale=1,
                        thickness=2,
                        color=(color[1], color[2], color[0]))
        cv2.putText(img,
                    text=str(num_of_object),
                    org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                    fontFace=3,
                    fontScale=1,
                    color=(255, 0, 0))
        img = img[:, :, ::-1]
        return img

    img_tensor = tf.squeeze(img_batch, 0)
    img_tensor_with_boxes = tf.py_func(draw_box_cv,
                                       inp=[img_tensor, boxes, scores],
                                       Tout=[tf.uint8])
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
    return img_tensor_with_boxes


def draw_boxes_with_categories_and_scores(img_batch, boxes, labels, scores):

    def draw_box_cv(img, boxes, labels, scores):
        img = img + np.array(cfgs.PIXEL_MEAN)
        boxes = boxes.astype(np.int64)
        labels = labels.astype(np.int32)
        img = np.array(img*255/np.max(img), np.uint8)

        num_of_object = 0
        for i, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]

            label = labels[i]
            score = scores[i]
            if label != 0:
                num_of_object += 1
                color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
                cv2.rectangle(img,
                              pt1=(xmin, ymin),
                              pt2=(xmax, ymax),
                              color=color,
                              thickness=2)
                cv2.rectangle(img,
                              pt1=(xmin, ymin),
                              pt2=(xmin+120, ymin+15),
                              color=color,
                              thickness=-1)
                category = LABEl_NAME_MAP[label]
                cv2.putText(img,
                            text=category+": "+str(score),
                            org=(xmin, ymin+10),
                            fontFace=1,
                            fontScale=1,
                            thickness=2,
                            color=(color[1], color[2], color[0]))
        cv2.putText(img,
                    text=str(num_of_object),
                    org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                    fontFace=3,
                    fontScale=1,
                    color=(255, 0, 0))
        img = img[:, :, ::-1]
        return img

    img_tensor = tf.squeeze(img_batch, 0)
    img_tensor_with_boxes = tf.py_func(draw_box_cv,
                                       inp=[img_tensor, boxes, labels, scores],
                                       Tout=[tf.uint8])
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
    return img_tensor_with_boxes


def draw_boxes_with_categories_and_scores_rotate(img_batch, boxes, labels, scores):

    def draw_box_cv(img, boxes, labels, scores):
        img = img + np.array(cfgs.PIXEL_MEAN)
        boxes = boxes.astype(np.int64)
        labels = labels.astype(np.int32)
        img = np.array(img*255/np.max(img), np.uint8)

        num_of_object = 0
        for i, box in enumerate(boxes):

            x_c, y_c, w, h, theta = box[0], box[1], box[2], box[3], box[4]
            label = labels[i]
            score = scores[i]
            if label != 0:
                num_of_object += 1

                rect = ((x_c, y_c), (w, h), theta)
                rect = cv2.boxPoints(rect)
                rect = np.int0(rect)
                color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
                cv2.drawContours(img, [rect], -1, color, 3)

                cv2.rectangle(img,
                              pt1=(x_c, y_c),
                              pt2=(x_c+120, y_c+15),
                              color=color,
                              thickness=-1)
                category = LABEl_NAME_MAP[label]
                cv2.putText(img,
                            text=category+": "+str(score),
                            org=(x_c, y_c+10),
                            fontFace=1,
                            fontScale=1,
                            thickness=2,
                            color=(color[1], color[2], color[0]))
        cv2.putText(img,
                    text=str(num_of_object),
                    org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                    fontFace=3,
                    fontScale=1,
                    color=(255, 0, 0))
        img = img[:, :, ::-1]
        return img

    img_tensor = tf.squeeze(img_batch, 0)
    img_tensor_with_boxes = tf.py_func(draw_box_cv,
                                       inp=[img_tensor, boxes, labels, scores],
                                       Tout=[tf.uint8])
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
    return img_tensor_with_boxes

def draw_boxes_rotate_my(img_batch, boxes, labels, scores):

    def draw_box_cv(img, boxes, labels, scores):
        img = img + np.array(cfgs.PIXEL_MEAN)
        labels = labels.astype(np.int32)
        img = np.array(img*255/np.max(img), np.uint8)

        num_of_object = 0
        for i, box in enumerate(boxes):
            box = np.array(box)
            x1, y1, x2, y2, h = box[0], box[1], box[2], box[3], box[4]
            # get four points
            if abs(x1 - x2) < 1e-6:
                x3 = x2 + h
                y3 = y2
            else:
                k1 = (y2 - y1) / (x2 - x1)
                if abs(k1) < 0.01:  # tan(pi/180)
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

            label = labels[i]
            score = scores[i]
            if label != 0:
                num_of_object += 1
                tmp = np.int0([x1, y1, x2, y2, x3, y3, x4, y4]).reshape([4, 2])
                rect2 = cv2.minAreaRect(tmp)
                x_c, y_c = np.int0(rect2[0][0]), np.int0(rect2[0][1])

                rect = cv2.boxPoints(rect2)
                rect = np.int0(rect)
                #color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
                color = (0, 0, 255)
                cv2.drawContours(img, [rect], -1, color, 2)
                cv2.line(img, (np.int0(x1), np.int0(y1)), (np.int0(x2), np.int0(y2)), (0,255,0), 2)
                cv2.circle(img, (np.int0(x1), np.int0(y1)), 1, (0,255,0), 8)

                cv2.rectangle(img,
                              pt1=(x_c, y_c),
                              pt2=(x_c+120, y_c+15),
                              color=color,
                              thickness=-1)
                category = LABEl_NAME_MAP[label]
                cv2.putText(img,
                            text=category+": "+str(score),
                            org=(x_c, y_c+10),
                            fontFace=1,
                            fontScale=1,
                            thickness=2,
                            color=(color[1], color[2], color[0]))
        cv2.putText(img,
                    text=str(num_of_object),
                    org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                    fontFace=3,
                    fontScale=1,
                    color=(255, 0, 0))
        img = img[:, :, ::-1]
        return img

    img_tensor = tf.squeeze(img_batch, 0)
    img_tensor_with_boxes = tf.py_func(draw_box_cv,
                                       inp=[img_tensor, boxes, labels, scores],
                                       Tout=[tf.uint8])
    img_tensor_with_boxes = tf.reshape(img_tensor_with_boxes, tf.shape(img_batch))
    return img_tensor_with_boxes



if __name__ == "__main__":
    print (1)

