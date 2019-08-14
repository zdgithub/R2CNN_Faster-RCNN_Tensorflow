import os
import numpy as np
import cv2
import scipy.io as sio
import math
from sklearn import preprocessing
from sklearn.externals import joblib
from libs.configs import cfgs


def draw_r2cnn_box(img, boxes, labels, scores):
    img = img + np.array(cfgs.PIXEL_MEAN)
    labels = labels.astype(np.int32)
    img = np.array(img * 255 / np.max(img), np.uint8)

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
        if label != 0:
            num_of_object += 1
            tmp = np.int0([x1, y1, x2, y2, x3, y3, x4, y4]).reshape([4, 2])
            rect2 = cv2.minAreaRect(tmp)
            x_c, y_c = np.int0(rect2[0][0]), np.int0(rect2[0][1])

            rect = cv2.boxPoints(rect2)
            rect = np.int0(rect)

            color = (0, 0, 255)
            cv2.drawContours(img, [rect], -1, color, 1)
            cv2.line(img, (np.int0(x1), np.int0(y1)), (np.int0(x2), np.int0(y2)), (255, 0, 0), 1)
            cv2.circle(img, (np.int0(x1), np.int0(y1)), 1, (255, 0, 0), 4)

            # if scores[i] >= 0.9:
            #     cv2.drawContours(img, [rect], -1, (0, 0, 255), 1)  # red
            # elif scores[i] >= 0.8:
            #     cv2.drawContours(img, [rect], -1, (255, 0, 0), 1)  # blue
            # else:
            #     cv2.drawContours(img, [rect], -1, (0, 255, 0), 1)  # green

            if scores is not None:
                cv2.rectangle(img,
                              pt1=(x_c, y_c),
                              pt2=(x_c + 120, y_c + 15),
                              color=(0,0,255),
                              thickness=-1)
                cv2.putText(img,
                            text=str(scores[i]),
                            org=(x_c, y_c+10),
                            fontFace=1,
                            fontScale=1,
                            thickness=2,
                            color=(0,255,0))

    cv2.putText(img,
                text=str(num_of_object),
                org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                fontFace=3,
                fontScale=3,
                color=(0, 255, 255))
    return img


def draw_rois_scores(img, boxes, scores):
    img = img + np.array(cfgs.PIXEL_MEAN)
    boxes = boxes.astype(np.int64)
    img = np.array(img * 255 / np.max(img), np.uint8)

    num_of_object = 0
    for i, box in enumerate(boxes):
        xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]

        score = scores[i]
        num_of_object += 1
        cv2.rectangle(img,
                      pt1=(xmin, ymin),
                      pt2=(xmax, ymax),
                      color=(0, 0, 255),
                      thickness=2)
        """cv2.rectangle(img,
                      pt1=(xmin, ymin),
                      pt2=(xmin + 120, ymin + 15),
                      color=(0,0,255),
                      thickness=-1)
        cv2.putText(img,
                    text=str(score),
                    org=(xmin, ymin + 10),
                    fontFace=1,
                    fontScale=1,
                    thickness=2,
                    color=(255,0,0))"""

    cv2.putText(img,
                text=str(num_of_object),
                org=((img.shape[1]) // 2, (img.shape[0]) // 2),
                fontFace=3,
                fontScale=3,
                color=(0, 255, 0))
    return img


def getRboxDegree(contours, angles):
    n = contours.shape[0]
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
        theta = angles[i]
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

        from_idx = i  # the raw index of r2cnn detected final boxes
        dtbox[i, :] = [x0, y0, x1, y1, x2, y2, x3, y3, dg, ycenter, from_idx]

    dtbox = dtbox[dtbox[:, 9].argsort()]  # sorted by ycenter

    return dtbox


def geneTestImageFeats(dtbox):
    # dtbox = [x0, y0, x1, y1, x2, y2, x3, y3, dg, ycenter, from_idx]
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
        xcenter = (x0 + x2) / 2.0
        ycenter = (y0 + y2) / 2.0
        a = math.sqrt((x0 - x1) ** 2 + (y0 - y1) ** 2)
        b = math.sqrt((x0 - x3) ** 2 + (y0 - y3) ** 2)
        if x0 < xcenter:
            Sh = a
            Sw = b
        else:
            Sh = b
            Sw = a

        features[i, :] = [Sh, Sw, theta, xcenter, ycenter]

    num = n
    fuv = []
    # ignore first and last bones
    for j in range(1, num - 1):
        up = j - 1
        down = j + 1
        v1 = features[up, :]  # 1dim
        v2 = features[down, :]
        u = features[j, :]
        f1 = np.concatenate((v1 - u, u - v2))
        f2 = np.concatenate((f1, u))
        fuv.append(f2)

    fuv = np.array(fuv).reshape((-1, 15))  # n-2
    # print('fuv shape:', fuv.shape)  (18, 15)

    if fuv.shape[0] > 0:
        # column values to (0, 1)
        min_max_scaler = preprocessing.MinMaxScaler()
        fuv_minmax = min_max_scaler.fit_transform(fuv)
    else:
        fuv_minmax = fuv

    return fuv_minmax, features


def svmPred(fuv, features):
    # feature = [Sh, Sw, theta, xcenter, ycenter]  # n
    clf = joblib.load('trained_theta.pkl')
    pred = clf.decision_function(fuv)

    n = pred.shape[0]
    final_bones = np.ones(n + 2)  # preserve the first and end bones
    final_bones[0] = 1  # 0
    final_bones[-1] = 1  # 0

    idx = list(range(1, n + 1))
    decision = np.zeros((n, 2))
    decision[:, 0] = idx
    decision[:, 1] = pred

    decision = decision[decision[:, 1].argsort()]  #
    # delete svm prediction score < -1.45 bones
    for k in range(n):
        j = int(decision[k, 0])
        pre = decision[k, 1]
        if pre < -0.99:  # -1.2 -1.1
            curH = features[j, 0]
            preH = features[j - 1, 0]
            curY = features[j, 4]
            preY = features[j - 1, 4]
            diff = abs(curY - preY) - (curH + preH) * 3  # if two bones are too away from each other
            if diff < 0:
                final_bones[j] = 0

    # return index of zero elements
    dtbox_index = np.nonzero(final_bones == 0)[0]  # (len,)

    return dtbox_index


def nmsRotate(boxes, scores, iou_threshold, max_output_size):
    keep = []

    order = scores.argsort()[::-1]
    num = boxes.shape[0]

    suppressed = np.zeros((num), dtype=np.int)

    for _i in range(num):
        if len(keep) >= max_output_size:
            break

        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        r1 = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), boxes[i, 4])
        area_r1 = boxes[i, 2] * boxes[i, 3]
        for _j in range(_i + 1, num):
            j = order[_j]
            if suppressed[i] == 1:
                continue
            r2 = ((boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), boxes[j, 4])
            area_r2 = boxes[j, 2] * boxes[j, 3]
            inter = 0.0

            int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]
            if int_pts is not None:
                order_pts = cv2.convexHull(int_pts, returnPoints=True)

                int_area = cv2.contourArea(order_pts)

                inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + 1e-5)

            if inter >= iou_threshold:
                suppressed[j] = 1

    return np.array(keep, np.int64)


def draw_rotate_box_cv_my(boxes, labels):
    boxes = boxes.astype(np.int64)
    labels = labels.astype(np.int32)

    num_of_object = 0

    contours = []
    angles = []

    for i, box in enumerate(boxes):
        x_c, y_c, w, h, theta = box[0], box[1], box[2], box[3], box[4]

        label = labels[i]
        if label != 0:
            num_of_object += 1
            # color = (np.random.randint(255), np.random.randint(255), np.random.randint(255))
            color = (0, 0, 255)
            rect = ((x_c, y_c), (w, h), theta)
            rect = cv2.boxPoints(rect)
            contours.append(rect)
            angles.append(theta)

    contours = np.array(contours)
    angles = np.array(angles)

    return contours, angles


def draw_contour_box(img, contours, angles, scores):
    img = img + np.array(cfgs.PIXEL_MEAN)
    img = np.array(img, np.float32)
    img = np.array(img * 255 / np.max(img), np.uint8)

    n = contours.shape[0]
    delete_idx = []
    delete_idx.append(0)
    delete_idx.append(n - 1)

    for i, rect in enumerate(contours):
        if (i == 0) or (i == (n - 1)):
            continue
        rect = np.int0(rect)
        if scores[i] >= 0.98:
            cv2.drawContours(img, [rect], -1, (0, 0, 255), 1)  # red
        elif scores[i] >= 0.9:
            cv2.drawContours(img, [rect], -1, (255, 0, 0), 1)  # blue
        else:
            cv2.drawContours(img, [rect], -1, (0, 255, 0), 1)  # green

    # contours = np.array(contours)
    # angles = np.array(angles)
    contours = np.delete(contours, delete_idx, axis=0)
    angles = np.delete(angles, delete_idx, axis=0)

    return img, contours, angles


def getCobb(finalbox, img_array):
    # finalbox = [x0, y0, x1, y1, x2, y2, x3, y3, degree, ycenter, from_idx]  dtbox
    degrees = finalbox[:, 8]  # (len,)
    n = finalbox.shape[0]
    deg = degrees.copy()

    # img_array.astype(np.float32)
    # img = np.array(img_array * 255 / np.max(img_array), dtype=np.uint8)
    img = img_array.copy()

    # draw degree of each detected bone
    for i in range(n):
        xcen = (finalbox[i, 0] + finalbox[i, 4]) / 2.0
        ycen = (finalbox[i, 1] + finalbox[i, 5]) / 2.0
        cv2.putText(img, text=str(int(deg[i])), org=(int(xcen), int(ycen + 10)), fontFace=1, fontScale=2, thickness=2,
                    color=(0, 0, 255))

    # find turn points
    tmp = [0, deg[0]]
    res = []
    for i in range(1, n):
        if deg[i] * tmp[1] >= 0:  # same symbol
            if abs(deg[i]) > abs(tmp[1]):
                tmp = [i, deg[i]]
        elif deg[i] * tmp[1] < 0:
            res.append(tmp)
            tmp = [i, deg[i]]
    res.append(tmp)
    res = np.array(res)

    # all bones' degrees are the same symbol
    cur = [0, deg[0]]
    if res.shape[0] == 1:
        for i in range(1, n):
            if abs(deg[i]) < abs(cur[1]):
                cur = [i, deg[i]]
        res = np.concatenate((np.array(cur).reshape(-1, 2), res))

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
            p2 = np.array([x1, y1])

            # h = math.sqrt((x0-x3)**2 + (y0-y3)**2)
            # w = math.sqrt((x0-x1)**2 + (y0-y1)**2)
            # if h > w*1.4:
            #    p2 = np.array([x3, y3])
            # else:
            #    p2 = np.array([x1, y1])
        # line end points
        p3 = (2.5 * p2 - 1.5 * p1)
        p4 = (2.5 * p1 - 1.5 * p2)
        if p3[0] < p4[0]:
            lines.append([p3[0], p3[1], p4[0], p4[1], theta, idx])
        else:
            lines.append([p4[0], p4[1], p3[0], p3[1], theta, idx])

    lines = np.array(lines)

    cobb = []
    for i in range(lines.shape[0]):
        if i >= 1:
            b1 = int(lines[i - 1, 5])
            b2 = int(lines[i, 5])
            diff_idx = abs(b1 - b2)  # diff number of the two bones
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
                    b2h = math.sqrt((b2_x0 - b2_x1) ** 2 + (b2_y0 - b2_y1) ** 2)
                else:
                    b2h = math.sqrt((b2_x0 - b2_x3) ** 2 + (b2_y0 - b2_y3) ** 2)
                # distance of two lines' center = two bones' center
                b1_c = [(lines[i - 1, 0] + lines[i - 1, 2]) / 2.0, (lines[i - 1, 1] + lines[i - 1, 3]) / 2.0]
                b2_c = [(lines[i, 0] + lines[i, 2]) / 2.0, (lines[i, 1] + lines[i, 3]) / 2.0]
                dist = math.sqrt((b1_c[0] - b2_c[0]) ** 2 + (b1_c[1] - b2_c[1]) ** 2)
                #
                if diff_idx == 1 and dist < b2h * 1.5:
                    continue
                elif diff_idx == 2 and dist < b2h * 2.5:
                    continue

            ang = abs(lines[i, 4] - lines[i - 1, 4])
            cobb.append(ang)
            up = lines[i - 1, :]
            up = up.astype(np.int32)
            cv2.line(img, (up[0], up[1]), (up[2], up[3]), (0, 255, 255), 2)
            down = lines[i, :]
            down = down.astype(np.int32)
            cv2.line(img, (down[0], down[1]), (down[2], down[3]), (0, 255, 255), 2)
            linx = (lines[i, 0] + lines[i - 1, 0]) / 2.0
            liny = (lines[i, 1] + lines[i - 1, 1]) / 2.0
            cv2.putText(img, text=str(int(ang)), org=(int(linx), int(liny)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=2, thickness=2, color=(0, 255, 255))

    return img
