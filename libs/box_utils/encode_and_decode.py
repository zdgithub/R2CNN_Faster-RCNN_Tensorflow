# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import math



def decode_boxes(encode_boxes, reference_boxes, scale_factors=None):
    '''

    :param encoded_boxes:[N, 4]
    :param reference_boxes: [N, 4] .
    :param scale_factors: use for scale.

    in the first stage, reference_boxes  are anchors
    in the second stage, reference boxes are proposals(decode) produced by first stage
    :return:decode boxes [N, 4]
    '''

    t_xcenter, t_ycenter, t_w, t_h = tf.unstack(encode_boxes, axis=1)
    if scale_factors:
        t_xcenter /= scale_factors[0]
        t_ycenter /= scale_factors[1]
        t_w /= scale_factors[2]
        t_h /= scale_factors[3]

    reference_xmin, reference_ymin, reference_xmax, reference_ymax = tf.unstack(reference_boxes, axis=1)
    # reference boxes are anchors in the first stage

    reference_xcenter = (reference_xmin + reference_xmax) / 2.
    reference_ycenter = (reference_ymin + reference_ymax) / 2.
    reference_w = reference_xmax - reference_xmin
    reference_h = reference_ymax - reference_ymin

    predict_xcenter = t_xcenter * reference_w + reference_xcenter
    predict_ycenter = t_ycenter * reference_h + reference_ycenter
    predict_w = tf.exp(t_w) * reference_w
    predict_h = tf.exp(t_h) * reference_h

    predict_xmin = predict_xcenter - predict_w / 2.
    predict_xmax = predict_xcenter + predict_w / 2.
    predict_ymin = predict_ycenter - predict_h / 2.
    predict_ymax = predict_ycenter + predict_h / 2.

    return tf.transpose(tf.stack([predict_xmin, predict_ymin,
                                  predict_xmax, predict_ymax]))


def encode_boxes(unencode_boxes, reference_boxes, scale_factors=None):
    '''

    :param unencode_boxes: [-1, 4]
    :param reference_boxes: [-1, 4]
    :return: encode_boxes [-1, 4]
    '''

    xmin, ymin, xmax, ymax = unencode_boxes[:, 0], unencode_boxes[:, 1], unencode_boxes[:, 2], unencode_boxes[:, 3]

    reference_xmin, reference_ymin, reference_xmax, reference_ymax = reference_boxes[:, 0], reference_boxes[:, 1], \
                                                                     reference_boxes[:, 2], reference_boxes[:, 3]

    x_center = (xmin + xmax) / 2.
    y_center = (ymin + ymax) / 2.
    w = xmax - xmin + 1e-8
    h = ymax - ymin + 1e-8

    reference_xcenter = (reference_xmin + reference_xmax) / 2.
    reference_ycenter = (reference_ymin + reference_ymax) / 2.
    reference_w = reference_xmax - reference_xmin + 1e-8
    reference_h = reference_ymax - reference_ymin + 1e-8

    # w + 1e-8 to avoid NaN in division and log below

    t_xcenter = (x_center - reference_xcenter) / reference_w
    t_ycenter = (y_center - reference_ycenter) / reference_h
    t_w = np.log(w/reference_w)
    t_h = np.log(h/reference_h)

    if scale_factors:
        t_xcenter *= scale_factors[0]
        t_ycenter *= scale_factors[1]
        t_w *= scale_factors[2]
        t_h *= scale_factors[3]

    return np.transpose(np.stack([t_xcenter, t_ycenter, t_w, t_h], axis=0))


# def decode_boxes_rotate(encode_boxes, reference_boxes, scale_factors=None):
#     '''
#
#     :param encode_boxes:[N, 5]
#     :param reference_boxes: [N, 5] .
#     :param scale_factors: use for scale
#     in the rpn stage, reference_boxes are anchors
#     in the fast_rcnn stage, reference boxes are proposals(decode) produced by rpn stage
#     :return:decode boxes [N, 5]
#     '''
#
#     t_xcenter, t_ycenter, t_w, t_h, t_theta = tf.unstack(encode_boxes, axis=1)
#     if scale_factors:
#         t_xcenter /= scale_factors[0]
#         t_ycenter /= scale_factors[1]
#         t_w /= scale_factors[2]
#         t_h /= scale_factors[3]
#         t_theta /= scale_factors[4]
#     reference_xmin, reference_ymin, reference_xmax, reference_ymax = tf.unstack(reference_boxes, axis=1)
#     reference_x_center = (reference_xmin + reference_xmax) / 2.
#     reference_y_center = (reference_ymin + reference_ymax) / 2.
#     reference_w = reference_xmax - reference_xmin
#     reference_h = reference_ymax - reference_ymin
#     reference_theta = tf.ones(tf.shape(reference_xmin)) * -90
#     predict_x_center = t_xcenter * reference_w + reference_x_center
#     predict_y_center = t_ycenter * reference_h + reference_y_center
#     predict_w = tf.exp(t_w) * reference_w
#     predict_h = tf.exp(t_h) * reference_h
#     predict_theta = t_theta * 180 / math.pi + reference_theta
#
#     decode_boxes = tf.transpose(tf.stack([predict_x_center, predict_y_center,
#                                           predict_w, predict_h, predict_theta]))
#     return decode_boxes


def decode_boxes_rotate(encode_boxes, reference_boxes, scale_factors=None):
    '''

    :param encode_boxes:[N, 5]
    :param reference_boxes: [N, 5] .
    :param scale_factors: use for scale
    in the rpn stage, reference_boxes are anchors
    in the fast_rcnn stage, reference boxes are proposals(decode) produced by rpn stage
    :return:decode boxes [N, 5]
    '''

    t_x1, t_y1, t_x2, t_y2, t_h = tf.unstack(encode_boxes, axis=1)
    if scale_factors:
        t_x1 /= scale_factors[0]
        t_y1 /= scale_factors[1]
        t_x2 /= scale_factors[2]
        t_y2 /= scale_factors[3]
        t_h /= scale_factors[4]
    reference_xmin, reference_ymin, reference_xmax, reference_ymax = tf.unstack(reference_boxes, axis=1)
    reference_x1 = reference_xmin
    reference_y1 = reference_ymin
    reference_x2 = reference_xmax
    reference_y2 = reference_ymin
    reference_w = reference_xmax - reference_xmin
    reference_h = reference_ymax - reference_ymin

    predict_x1 = t_x1 * reference_w + reference_x1
    predict_y1 = t_y1 * reference_h + reference_y1
    predict_x2 = t_x2 * reference_w + reference_x2
    predict_y2 = t_y2 * reference_h + reference_y2
    predict_h = tf.exp(t_h) * reference_h

    decode_boxes = tf.transpose(tf.stack([predict_x1, predict_y1, predict_x2, predict_y2, predict_h]))

    return decode_boxes



# def encode_boxes_rotate(unencode_boxes, reference_boxes, scale_factors=None):
#     '''
#     :param unencode_boxes: [batch_size*H*W*num_anchors_per_location, 5]
#     :param reference_boxes: [H*W*num_anchors_per_location, 4]
#     :return: encode_boxes [-1, 5]
#     '''
#     x_center, y_center, w, h, theta= \
#         unencode_boxes[:, 0], unencode_boxes[:, 1], unencode_boxes[:, 2], unencode_boxes[:, 3], unencode_boxes[:, 4]
#     reference_xmin, reference_ymin, reference_xmax, reference_ymax = \
#         reference_boxes[:, 0], reference_boxes[:, 1], reference_boxes[:, 2], reference_boxes[:, 3]
#     reference_x_center = (reference_xmin + reference_xmax) / 2.
#     reference_y_center = (reference_ymin + reference_ymax) / 2.
#     # here maybe have logical error, reference_w and reference_h should exchange,
#     # but it doesn't seem to affect the result.
#     reference_w = reference_xmax - reference_xmin
#     reference_h = reference_ymax - reference_ymin
#     reference_theta = np.ones(reference_xmin.shape) * -90
#
#     reference_w += 1e-8
#     reference_h += 1e-8
#     w += 1e-8
#     h += 1e-8  # to avoid NaN in division and log below
#     t_xcenter = (x_center - reference_x_center) / reference_w
#     t_ycenter = (y_center - reference_y_center) / reference_h
#     t_w = np.log(w / reference_w)
#     t_h = np.log(h / reference_h)
#     t_theta = (theta - reference_theta) * math.pi / 180
#     if scale_factors:
#         t_xcenter *= scale_factors[0]
#         t_ycenter *= scale_factors[1]
#         t_w *= scale_factors[2]
#         t_h *= scale_factors[3]
#         t_theta *= scale_factors[4]
#     return np.transpose(np.stack([t_xcenter, t_ycenter, t_w, t_h, t_theta]))


# my r2cnn encode_boxes_rotate
# to get regression targets, so np not tf
def encode_boxes_rotate(unencode_boxes, reference_boxes, scale_factors=None):
    '''
    :param unencode_boxes: [batch_size*H*W*num_anchors_per_location, 5]  gt
    :param reference_boxes: [H*W*num_anchors_per_location, 4]   roi
    :return: encode_boxes [-1, 5]
    '''
    x1, y1, x2, y2, h = \
        unencode_boxes[:, 0], unencode_boxes[:, 1], unencode_boxes[:, 2], unencode_boxes[:, 3], unencode_boxes[:, 4]
    reference_xmin, reference_ymin, reference_xmax, reference_ymax = \
        reference_boxes[:, 0], reference_boxes[:, 1], reference_boxes[:, 2], reference_boxes[:, 3]
    reference_x1 = reference_xmin
    reference_y1 = reference_ymin
    reference_x2 = reference_xmax
    reference_y2 = reference_ymin
    reference_h = reference_ymax - reference_ymin + 1e-8  # to avoid NaN in division and log below
    reference_w = reference_xmax - reference_xmin + 1e-8

    t_x1 = (x1 - reference_x1) / reference_w
    t_y1 = (y1 - reference_y1) / reference_h
    t_x2 = (x2 - reference_x2) / reference_w
    t_y2 = (y2 - reference_y2) / reference_h
    t_h = np.log(h / reference_h)

    if scale_factors:
        t_x1 *= scale_factors[0]
        t_y1 *= scale_factors[1]
        t_x2 *= scale_factors[2]
        t_y2 *= scale_factors[3]
        t_h *= scale_factors[4]

    return np.transpose(np.stack([t_x1, t_y1, t_x2, t_y2, t_h]))