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

from data.io.image_preprocess import short_side_resize_for_inference_data
from libs.configs import cfgs
from libs.networks import build_whole_network_my
from help_utils.tools import *
from libs.box_utils import draw_box_in_img, show_box_in_tensor
from help_utils import tools
from libs.box_utils import coordinate_convert
import mylibs


def inference(det_net, data_dir):

    # 1. preprocess img
    img_plac = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3])
    img_batch = tf.cast(img_plac, tf.float32)
    img_batch = img_batch - tf.constant(cfgs.PIXEL_MEAN)
    img_batch = short_side_resize_for_inference_data(img_tensor=img_batch,
                                                     target_shortside_len=cfgs.IMG_SHORT_SIDE_LEN)

    rois, roi_scores, det_boxes_h, det_scores_h, det_category_h, \
    all_boxes_r, all_scores_r, all_category_r = det_net.build_whole_detection_network(input_img_batch=img_batch,
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
        for i, a_img_name in enumerate(imgs):

            raw_img = cv2.imread(os.path.join(data_dir,
                                              a_img_name))
            # raw_h, raw_w = raw_img.shape[0], raw_img.shape[1]

            start = time.time()
            resized_img, rois_, roi_scores_, det_boxes_h_, det_scores_h_, det_category_h_, \
            all_boxes_r_, all_scores_r_, all_category_r_ = \
                sess.run(
                    [img_batch, rois, roi_scores, det_boxes_h, det_scores_h, det_category_h,
                     all_boxes_r, all_scores_r, all_category_r],
                    feed_dict={img_plac: raw_img}
                )
            end = time.time()

	    print('all rois shape:', rois_.shape)

            all_boxes_new = all_boxes_r_        # [-1, 5]
            all_scores_new = all_scores_r_      # [-1]
            all_category_new = all_category_r_  # [-1]
            
            print('all dets shape:', all_boxes_new.shape)

	    # draw all rois from proposals
	    #rois_img_all = mylibs.draw_rois_scores(np.squeeze(resized_img, 0), rois_, roi_scores_)
	    #score_gre_05 = np.reshape(np.where(np.greater_equal(roi_scores_, 0.5)), -1)
	    #score_gre_05_rois = rois_[score_gre_05]
 	    #score_gre_05_scores = roi_scores_[score_gre_05]
	    #rois_img_part = mylibs.draw_rois_scores(np.squeeze(resized_img, 0), score_gre_05_rois, score_gre_05_scores)

            	     
            # draw all 800 detection boxes
	    all_indices = np.reshape(np.where(np.greater_equal(all_scores_new, cfgs.SHOW_SCORE_THRSHOLD)), -1)
	    left_boxes = all_boxes_new[all_indices]
	    left_scores = all_scores_new[all_indices]
	    left_category = all_category_new[all_indices]
	    #print('greater than score shape:', left_boxes.shape)
	    detection_r = draw_box_in_img.draw_rotate_box_cv(np.squeeze(resized_img, 0),
							     boxes=left_boxes,
   							     labels=left_category,
							     scores=left_scores,
							     imgname=a_img_name)
	    

            """
            while True:
                # nms
                keep = mylibs.nmsRotate(all_boxes_new, all_scores_new,
                                        cfgs.FAST_RCNN_NMS_IOU_THRESHOLD, cfgs.FAST_RCNN_NMS_MAX_BOXES_PER_CLASS)

                final_boxes = all_boxes_new[keep]
                final_scores = all_scores_new[keep]
                final_category = all_category_new[keep]

                kept_indices = np.reshape(np.where(np.greater_equal(final_scores, cfgs.SHOW_SCORE_THRSHOLD)), -1)
                det_boxes_new = final_boxes[kept_indices]
                det_scores_new = final_scores[kept_indices]
                det_category_new = final_category[kept_indices]
                # detected boxes
                contours, angles = mylibs.draw_rotate_box_cv_my(det_boxes_new, det_category_new)
                dtbox = mylibs.getRboxDegree(contours, angles)  # n
                #print('dtbox shape is', dtbox.shape)
                fuv, features = mylibs.geneTestImageFeats(dtbox)          # n-2
                if fuv.shape[0] == 0:
                    print(a_img_name, 'left none bones')
                    break

                dtbox_idx = mylibs.svmPred(fuv, features)  # deleted bones index
                #print('dtbox_idx shape is', dtbox_idx.shape)

                contour_idx = np.reshape(dtbox[dtbox_idx][:, -1], -1).astype(np.int32)
                kept_idx = kept_indices[contour_idx]
                keep_index = keep[kept_idx]
                #print('keep_index shape is', keep_index.shape)

                #n = det_boxes_new.shape[0]
                if len(dtbox_idx) == 0:   # no svm deleted bones
                    break
                else:
                    # delete some rows
                    all_boxes_new = np.delete(all_boxes_new, keep_index, axis=0)
                    all_scores_new = np.delete(all_scores_new, keep_index, axis=0)
                    all_category_new = np.delete(all_category_new, keep_index, axis=0)

            # unsorted
            fcontours = contours
            fangles = angles
            fscores = det_scores_new
   
  	    """
            """
            # save contours and angles to showSVMdecision scores
  	    detection_r = draw_box_in_img.draw_rotate_box_cv(np.squeeze(resized_img, 0),
							     boxes=det_boxes_new,
							     labels=det_category_new,
							     scores=det_scores_new,
							     imgname=a_img_name)           
	    """
            """    
            # final_dtbox[i, :] = [x0, y0, x1, y1, x2, y2, x3, y3, dg, ycenter, from_idx]
            final_dtbox = mylibs.getRboxDegree(fcontours, fangles)
            dt_idx = final_dtbox[:, -1].astype(np.int32)  # sorted
            # sorted
            fcontours = fcontours[dt_idx]
            fangles = fangles[dt_idx]
            fscores = fscores[dt_idx]

            img, tcontours, tangles = mylibs.draw_contour_box(np.squeeze(resized_img, 0), fcontours, fangles, fscores)
            t_dtbox = mylibs.getRboxDegree(tcontours, tangles)

            cobb_img = mylibs.getCobb(t_dtbox, img)
            """
             
            save_dir = os.path.join(cfgs.INFERENCE_SAVE_PATH, cfgs.VERSION)
            tools.mkdir(save_dir)
            cv2.imwrite(save_dir + '/' + a_img_name + '_roi.jpg',
                        detection_r)



            view_bar('{} cost {}s'.format(a_img_name, (end - start)), i + 1, len(imgs))
            print()


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a R2CNN network')
    parser.add_argument('--data_dir', dest='data_dir',
                        help='data path',
                        default='./XrayPics/', type=str)
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

    det_net = build_whole_network_my.DetectionNetwork(base_network_name=cfgs.NET_NAME,
                                                   is_training=False)

    inference(det_net, data_dir=args.data_dir)

















