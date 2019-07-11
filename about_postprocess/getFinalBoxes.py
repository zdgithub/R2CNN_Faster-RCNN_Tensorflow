import os
import numpy as np
import scipy.io as sio
import math
import cv2

# Step 4
def getCobb(finalbox, img_array):
    # finalbox = [x0, y0, x1, y1, x2, y2, x3, y3, degree, ycenter, from_idx, idx_in_dtbox]
    degrees = finalbox[:, 8]  # (len,)
    n = finalbox.shape[0]
    deg = degrees.copy()

    # img_array.astype(np.float32)
    # img = np.array(img_array * 255 / np.max(img_array), dtype=np.uint8)
    img = img_array.copy()

    #
    for i in range(n):
        xcen = (finalbox[i, 0] + finalbox[i, 4]) / 2.0
        ycen = (finalbox[i, 1] + finalbox[i, 5]) / 2.0
        cv2.putText(img, text=str(int(deg[i])), org=(int(xcen), int(ycen+10)), fontFace=1, fontScale=2, thickness=2, color=(0,0,255))

    #
    tmp = [0, deg[0]]
    res = []
    for i in range(1, n):
        if deg[i] * tmp[1] >= 0:
            if abs(deg[i]) > abs(tmp[1]):
                tmp = [i, deg[i]]
        elif deg[i] * tmp[1] < 0:
            res.append(tmp)
            tmp = [i, deg[i]]
    res.append(tmp)
    res = np.array(res)

    #
    cur = [0, deg[0]]
    if res.shape[0] == 1:
        for i in range(1,n):
            if abs(deg[i]) < abs(cur[1]):
                cur = [i, deg[i]]
        res = np.concatenate((np.array(cur).reshape(-1,2), res))

    #
    m = res.shape[0]
    lines = []
    for i in range(m):
        idx = int(res[i, 0])
        theta = res[i, 1]
        x0 = finalbox[idx, 0]
        y0 = finalbox[idx, 1]
        x1 = finalbox[idx, 2]
        y1 = finalbox[idx, 3]
        x2 = finalbox[idx, 4]
        y2 = finalbox[idx, 5]
        x3 = finalbox[idx, 6]
        y3 = finalbox[idx, 7]
        xcenter = (x0 + x2) / 2.0
        p1 = np.array([x0, y0])
        if x0 < xcenter:
            p2 = np.array([x3, y3])
        else:
	    h = math.sqrt((x0-x3)**2 + (y0-y3)**2)
	    w = math.sqrt((x0-x1)**2 + (y0-y1)**2)
	    if h > w*1.4:
		p2 = np.array([x3, y3])
	    else:
                p2 = np.array([x1, y1])

        p3 = (3 * p2 - 2 * p1)
        p4 = (3 * p1 - 2 * p2)
        if p3[0] < p4[0]:
            lines.append([p3[0], p3[1], p4[0], p4[1], theta, idx])
        else:
            lines.append([p4[0], p4[1], p3[0], p3[1], theta, idx])
    lines = np.array(lines)

    cobb = []
    for i in range(lines.shape[0]):
        if i >= 1:
            b1 = int(lines[i-1, 5])
            b2 = int(lines[i, 5])
            diff_idx = finalbox[b2, -1] - finalbox[b1, -1]
            #
            if diff_idx <= 2:
                #
                b2_xcenter = (finalbox[b2, 0] + finalbox[b2, 4]) / 2.0
                b2_x0 = finalbox[b2, 0]
                b2_y0 = finalbox[b2, 1]
                b2_x1 = finalbox[b2, 2]
                b2_y1 = finalbox[b2, 3]
                b2_x3 = finalbox[b2, 6]
                b2_y3 = finalbox[b2, 7]
                if b2_x0 < b2_xcenter:
                    b2h = math.sqrt((b2_x0-b2_x1) ** 2 + (b2_y0-b2_y1) ** 2)
                else:
                    b2h = math.sqrt((b2_x0-b2_x3) ** 2 + (b2_y0-b2_y3) ** 2)
                #
                b1_c = [(lines[i - 1, 0] + lines[i - 1, 2]) / 2.0, (lines[i - 1, 1] + lines[i - 1, 3]) / 2.0]
                b2_c = [(lines[i, 0] + lines[i, 2]) / 2.0, (lines[i, 1] + lines[i, 3]) / 2.0]
                dist = math.sqrt((b1_c[0] - b2_c[0]) ** 2 + (b1_c[1] - b2_c[1]) ** 2)
                #
                if diff_idx == 1 and dist < b2h * 1.5:
                    continue
                elif diff_idx ==2 and dist < b2h * 2.5:
                    continue

            ang = abs(lines[i,4] - lines[i-1,4])
            cobb.append(ang)
            up = lines[i-1, :]
            up = up.astype(np.int32)
            cv2.line(img, (up[0], up[1]), (up[2], up[3]), (0, 255, 255), 2)
            down = lines[i, :]
            down = down.astype(np.int32)
            cv2.line(img, (down[0], down[1]), (down[2], down[3]), (0, 255, 255), 2)
            linx = (lines[i, 0] + lines[i-1, 0]) / 2.0
            liny = (lines[i, 1] + lines[i-1, 1]) / 2.0
            cv2.putText(img, text=str(int(ang)), org=(int(linx), int(liny)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, thickness=2, color=(0, 255, 255))

    return img


if __name__ == '__main__':
    p = '../inference_results/'
    files = os.listdir(p + 'R2CNN_rotatebox_2150_new')

    cobbsPath = p + 'Cobbs'
    if os.path.exists(cobbsPath):
        print('directory exists')
    else:
        os.mkdir(cobbsPath)
        print('Cobbs is created')

    for item in files:
        imgpath = os.path.join(p, 'R2CNN_rotatebox_2150_new', item)
        imgname = item[0:-10]
        img_array = cv2.imread(imgpath)

        dtbox = sio.loadmat(os.path.join(p, 'dtboxSet', imgname+'.mat'))['dtbox']  # array type
        pred = sio.loadmat(os.path.join(p, 'testpred', imgname+'-pred.mat'))['pred']  # array type
        n = pred.size
        final_bones = np.ones(n+2)  # don't save the first and end bones
        final_bones[0] = 0
        final_bones[-1] = 0

        idx = list(range(1, n+1))
        decision = np.zeros((n, 2))
        decision[:, 0] = idx
        decision[:, 1] = pred

        decision = decision[decision[:, 1].argsort()]  #
        # delete svm prediction score < -1.45 bones
        for k in range(n):
            j = int(decision[k, 0])
            pre = decision[k, 1]
            if pre < -1.45:
                final_bones[j] = 0

        # return index of nonzero elements
        index = np.nonzero(final_bones==1)[0]   #(len,)
        finalboxes = dtbox[index, :]
        finalboxes = np.concatenate((finalboxes, index.reshape(-1, 1)), 1)
        img = getCobb(finalboxes, img_array)


        savepath = os.path.join(p, 'Cobbs', imgname+'-cobb.jpg')
        cv2.imwrite(savepath, img)
    
    print('Successfully done.')

