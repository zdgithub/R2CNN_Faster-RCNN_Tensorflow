import os
import numpy as np
import scipy.io as sio
import copy
import math

# Step 1
# get contours' coordinates and angles of each picture
def getRboxDegree(p, imgname):
    contours = sio.loadmat(os.path.join(p, 'contours', imgname + '.jpg.mat'))['contours']  # array type
    angles = sio.loadmat(os.path.join(p, 'angles', imgname + '.jpg.mat'))['angles']  # array type
    n = contours.shape[0]
    # print('contours shape:', contours.shape) (20, 4, 2)
    # print('angels shape:', angles.shape) (1, 20)
    dtbox = np.zeros((n, 11))

    for i in range(n):
        a = contours[i, :, :]
        # print('a shape:', a.shape) (4, 2)
        x0 = a[0, 0]
        y0 = a[0, 1]
        x1 = a[1, 0]
        y1 = a[1, 1]
        x2 = a[2, 0]
        y2 = a[2, 1]
        x3 = a[3, 0]
        y3 = a[3, 1]
        theta = angles[0][i]
        # print(theta)
        xcenter = (x0 + x2) / 2.0
        ycenter = (y0 + y2) / 2.0
        if x0 < xcenter:
            if theta == -90:
                dg = 0
            else:
                dg = theta
        else:
            # h = math.sqrt((x0-x3)**2 + (y0-y3)**2)
            # w = math.sqrt((x0-x1)**2 + (y0-y1)**2)
            # if h > w*1.4:
            #      dg = theta
            # else:
            dg = 90 + theta
        # the raw index of r2cnn detected final boxes
        from_idx = i

        dtbox[i, :] = [x0, y0, x1, y1, x2, y2, x3, y3, dg, ycenter, from_idx]

    dtbox = dtbox[dtbox[:, 9].argsort()]  # sorted by ycenter
    savepath = os.path.join(p, 'dtboxSet', imgname + '.mat')
    sio.savemat(savepath, {'dtbox': dtbox})
    return


if __name__ == '__main__':
    p = '../inference_results/'
    files = os.listdir(p + 'R2CNN_rotatebox_2150_new')

    dtboxSetPath = p + 'dtboxSet'
    if os.path.exists(dtboxSetPath):
        print('directory exists')
    else:
        os.mkdir(dtboxSetPath)
        print('dtboxSet is created')

    for item in files:
        imgname = item[0:-10]
        getRboxDegree(p, imgname)
