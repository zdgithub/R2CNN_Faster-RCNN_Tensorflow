import random
import cv2
import os
import math
import numpy as np
from data.dataAugment.xml_helper import *


def rand_perturb(img_name, img_size, bboxes, out_xml_path):
    '''
    perturb a little randomly the coordinates of every ground truth box
    :return: perturbed annotation
    '''
    left_bboxes = []
    right_bboxes = []
    rot_bboxes = [[], []]
    for box in bboxes:
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        x3 = box[4]
        y3 = box[5]
        x4 = box[6]
        y4 = box[7]
        x_c = (x1 + x3) / 2
        y_c = (y1 + y3) / 2

        # crop ratio
        lrand = random.uniform(-0.05, 0.15)
        rrand = random.uniform(-0.05, 0.15)

        large_x1 = (1 + lrand) * x1 - lrand * x2
        large_y1 = (1 + lrand) * y1 - lrand * y2
        large_x4 = (1 + lrand) * x4 - lrand * x3
        large_y4 = (1 + lrand) * y4 - lrand * y3
        large_x2 = (1 + rrand) * x2 - rrand * x1
        large_y2 = (1 + rrand) * y2 - rrand * y1
        large_x3 = (1 + rrand) * x3 - rrand * x4
        large_y3 = (1 + rrand) * y3 - rrand * y4

        left_coords = [large_x1, large_y1, x2, y2, x3, y3, large_x4, large_y4]
        right_coords = [x1, y1, large_x2, large_y2, large_x3, large_y3, x4, y4]

        left_bboxes.append(left_coords)
        right_bboxes.append(right_coords)

        # rotate
        angles = [random.uniform(-2, -3), random.uniform(2, 3)]
        for i in range(2):
            angle = angles[i]
            rad = math.radians(angle)
            ang_x1 = (x1 - x_c) * math.cos(rad) + (y1 - y_c) * math.sin(rad) + x_c
            ang_x2 = (x2 - x_c) * math.cos(rad) + (y2 - y_c) * math.sin(rad) + x_c
            ang_x3 = (x3 - x_c) * math.cos(rad) + (y3 - y_c) * math.sin(rad) + x_c
            ang_x4 = (x4 - x_c) * math.cos(rad) + (y4 - y_c) * math.sin(rad) + x_c

            ang_y1 = (y1 - y_c) * math.cos(rad) - (x1 - x_c) * math.sin(rad) + y_c
            ang_y2 = (y2 - y_c) * math.cos(rad) - (x2 - x_c) * math.sin(rad) + y_c
            ang_y3 = (y3 - y_c) * math.cos(rad) - (x3 - x_c) * math.sin(rad) + y_c
            ang_y4 = (y4 - y_c) * math.cos(rad) - (x4 - x_c) * math.sin(rad) + y_c

            rot_coords=[ang_x1,ang_y1,ang_x2,ang_y2,ang_x3,ang_y3,ang_x4,ang_y4]
            rot_bboxes[i].append(rot_coords)


    generate_xml('left-'+img_name, left_bboxes, img_size, out_xml_path)
    generate_xml('right-' + img_name, right_bboxes, img_size, out_xml_path)
    generate_xml('rot1-' + img_name, rot_bboxes[0], img_size, out_xml_path)
    generate_xml('rot2-' + img_name, rot_bboxes[1], img_size, out_xml_path)





if __name__ == '__main__':

    src_img_path = r'E:\3AllRBox\VOCdevkit\ChabuduoJPEGImages'
    src_xml_path = r'E:\3AllRBox\VOCdevkit\ChabuduoAnnotation'

    out_img_path = r'E:\3AllRBox\VOCdevkit\RandJPEGImages'
    out_xml_path = r'E:\3AllRBox\VOCdevkit\RandAnnotation'

    for file in os.listdir(src_img_path):
        img_path = os.path.join(src_img_path, file)
        xml_path = os.path.join(src_xml_path, file[:-4] + '.xml')
        bboxes = read_xml(xml_path)  # [-1, 8]
        img = cv2.imread(img_path)
        cv2.imwrite(os.path.join(out_img_path, 'left-' + file), img)
        cv2.imwrite(os.path.join(out_img_path, 'right-' + file), img)
        cv2.imwrite(os.path.join(out_img_path, 'rot1-' + file), img)
        cv2.imwrite(os.path.join(out_img_path, 'rot2-' + file), img)
        rand_perturb(file, img.shape, bboxes, out_xml_path)














