import cv2
import os
import numpy as np


files = os.listdir('res')

for file in files:

    img = cv2.imread('res/'+file)
    # print('img shape is:', img.shape)
    # print('img min-max', np.min(img), np.max(img))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 1)

    save_path = os.path.join('contours', file)
    cv2.imwrite(save_path, img)

