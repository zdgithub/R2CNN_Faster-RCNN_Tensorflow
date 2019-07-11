import numpy as np
import os
import scipy.io as sio
import math
from sklearn import preprocessing

# Step 2
def geneTestImageFeats(p, imgname):
    # dtbox = [x0, y0, x1, y1, x2, y2, x3, y3, dg, ycenter, from_idx]
    dtbox = sio.loadmat(os.path.join(p, 'dtboxSet', imgname+'.mat'))['dtbox']
    n = dtbox.shape[0]
    features = np.zeros((n, 5))
    for i in range(n):
        x0 = dtbox[i, 0]
        y0 = dtbox[i, 1]
        x1 = dtbox[i, 2]
        y1 = dtbox[i, 3]
        x2 = dtbox[i, 4]
        y2 = dtbox[i, 5]
        x3 = dtbox[i, 6]
        y3 = dtbox[i, 7]
        theta = dtbox[i, 8]
        xcenter = (x0+x2) / 2.0
        ycenter = (y0+y2) / 2.0
        a = math.sqrt((x0-x1)**2 + (y0-y1)**2)
        b = math.sqrt((x0-x3)**2 + (y0-y3)**2)
        if x0 < xcenter:
            Sh = a
            Sw = b
        else:
            Sh = b
            Sw = a
        # else:
        #     if b > a*1.4:
        #     Sh = a
        #     Sw = b
        #     else:
		#     Sh = b
        #     Sw = a

        features[i, :] = [Sh, Sw, theta, xcenter, ycenter]

    return features


if __name__ == '__main__':
    p = '../inference_results/'
    files = os.listdir(p + 'dtboxSet')

    testfuvPath = p + 'testfuv'
    if os.path.exists(testfuvPath):
        print('directory exists')
    else:
        os.mkdir(testfuvPath)
        print('testfuv is created')

    for item in files:
        imgname = item[0:-4]
        feats = geneTestImageFeats(p, imgname)
        num = feats.shape[0]
        fuv = []
        # ignore first and last bones
        for j in range(1, num-1):
            up = j-1
            down = j+1
            v1 = feats[up, :]  # 1dim
            v2 = feats[down, :]
            u = feats[j, :]
            f1 = np.concatenate((v1-u, u-v2))
            f2 = np.concatenate((f1, u))
            fuv.append(f2)

        fuv = np.array(fuv)
        # print('fuv shape:', fuv.shape)  (18, 15)

        # column values to (0, 1)
        min_max_scaler = preprocessing.MinMaxScaler()
        fuv_minmax = min_max_scaler.fit_transform(fuv)

        savepath = os.path.join(p, 'testfuv', imgname+'-fuv.mat')
        sio.savemat(savepath, {'fuv':fuv_minmax})


