# -*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os, sys
sys.path.append("../")
import tensorflow as tf
import time
import cv2
import numpy as np
import argparse
import scipy.io as sio

from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.configs import cfgs
from libs.networks import build_whole_network
from help_utils.tools import *
from libs.box_utils import draw_box_in_img
from help_utils import tools
from libs.box_utils.coordinate_convert import my_getnms_area
import mylibs

def inference(det_net, data_dir):

    # 1. preprocess img
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])
    img_batch = tf.cast(img_plac, tf.float32)
    img_batch = img_batch - tf.constant(cfgs.PIXEL_MEAN)
    img_batch = short_side_resize_for_inference_data(img_tensor=img_batch,
                                                     target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN)

    det_boxes_h, det_scores_h, det_category_h, \
    det_boxes_r, det_scores_r, det_category_r = det_net.build_whole_detection_network(input_img_batch=img_batch,
                                                                                      gtboxes_h_batch=None,
                                                                                      gtboxes_r_batch=None)

    init_op = tf.group(
        tf.global_variables_initializer(),
        tf.local_variables_initializer()
    )

    restorer, restore_ckpt = det_net.get_restorer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(init_op)
        if not restorer is None:
            restorer.restore(sess, restore_ckpt)
            print('restore model')

        imgs = os.listdir(data_dir)
        save_dir = os.path.join(cfgs.INFERENCE_SAVE_PATH, cfgs.VERSION)
        tools.mkdir(save_dir)

        for i, a_img_name in enumerate(imgs):

            raw_img = cv2.imread(os.path.join(data_dir,
                                              a_img_name))
            # raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]

            start = time.time()
            resized_img, det_boxes_h_, det_scores_h_, det_category_h_, \
            det_boxes_r_, det_scores_r_, det_category_r_ = \
                sess.run(
                    [img_batch, det_boxes_h, det_scores_h, det_category_h,
                     det_boxes_r, det_scores_r, det_category_r],
                    feed_dict={img_plac: raw_img}
                )
            end = time.time()
            # det_boxes_r_ [x1, y1, x2, y2, h]

            det_detections_h = draw_box_in_img.draw_box_cv(np.squeeze(resized_img, 0),
                                                           boxes=det_boxes_h_,
                                                           labels=det_category_h_,
                                                           scores=det_scores_h_)

            det_detections_r = mylibs.draw_r2cnn_box(np.squeeze(resized_img, 0),
                                                     boxes=det_boxes_r_,
                                                     labels=det_category_r_,
                                                     scores=det_scores_r_)

            # draw angles
            deg_img, dtbox = mylibs.get_r2cnn_degree(det_detections_r, det_boxes_r_)

	    anglePath = cfgs.INFERENCE_SAVE_PATH + '/dtbox'
	    if not os.path.exists(anglePath):
		os.makedirs(anglePath)
	    savePath = os.path.join(anglePath, a_img_name+'.mat')
	    sio.savemat(savePath, {'dtbox':dtbox})

	    # draw cobb images
            #cobb_img = mylibs.get_r2cnn_cobb(deg_img, dtbox)

            cv2.imwrite(save_dir + '/' + a_img_name + '_r.jpg',
                        deg_img)
            view_bar('{} cost {}s'.format(a_img_name, (end - start)), i + 1, len(imgs))


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a R2CNN network')
    parser.add_argument('--data_dir', dest='data_dir',
                        help='data path',
                        default='./validSet/JPEGImages/', type=str)
    parser.add_argument('--gpu', dest='gpu',
                        help='gpu index',
                        default='0', type=str)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:')
    print(args)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    det_net = build_whole_network.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                   is_training=False)

    inference(det_net, data_dir=args.data_dir)

















