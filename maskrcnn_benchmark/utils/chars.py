import os

import cv2
import numpy as np


def char2num(char):
    if char in "0123456789":
        num = ord(char) - ord("0") + 1
    elif char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ":
        num = ord(char.lower()) - ord("a") + 11
    else:
        num = 0
    return num


def num2char(num):
    chars = "_0123456789abcdefghijklmnopqrstuvwxyz"
    char = chars[num]
    # if num >=1 and num <=10:
    # 	char = chr(ord('0') + num - 1)
    # elif num > 10 and num <= 36:
    # 	char = chr(ord('a') + num - 11)
    # else:
    # 	print('error number:%d'%(num))
    # 	exit()
    return char


def getstr_grid(seg, box, threshold=192):
    pos = 255 - (seg[0] * 255).astype(np.uint8)
    mask_index = np.argmax(seg, axis=0)
    mask_index = mask_index.astype(np.uint8)
    pos = pos.astype(np.uint8)
    string, score, rec_scores, char_polygons = seg2text(
        pos, mask_index, seg, box, threshold=threshold
    )
    return string, score, rec_scores, char_polygons


def seg2text(gray, mask, seg, box, threshold=192):
    ## input numpy
    img_h, img_w = gray.shape
    box_w = box[2] - box[0]
    box_h = box[3] - box[1]
    ratio_h = float(box_h) / img_h
    ratio_w = float(box_w) / img_w
    # SE1=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    # gray = cv2.erode(gray,SE1)
    # gray = cv2.dilate(gray,SE1)
    # gray = cv2.morphologyEx(gray,cv2.MORPH_CLOSE,SE1)
    ret, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    try:
        _, contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    chars = []
    scores = []
    char_polygons = []
    for i in range(len(contours)):
        char = {}
        temp = np.zeros((img_h, img_w)).astype(np.uint8)
        cv2.drawContours(temp, [contours[i]], 0, (255), -1)
        x, y, w, h = cv2.boundingRect(contours[i])
        c_x, c_y = x + w / 2, y + h / 2
        perimeter = cv2.arcLength(contours[i], True)
        epsilon = 0.01 * cv2.arcLength(contours[i], True)
        approx = cv2.approxPolyDP(contours[i], epsilon, True)
        pts = approx.reshape((-1, 2))
        pts[:, 0] = pts[:, 0] * ratio_w + box[0]
        pts[:, 1] = pts[:, 1] * ratio_h + box[1]
        polygon = list(pts.reshape((-1,)))
        polygon = list(map(int, polygon))
        if len(polygon) >= 6:
            char_polygons.append(polygon)
        # x1 = x * ratio_w + box[0]
        # y1 = y * ratio_h + box[1]
        # x3 = (x + w) * ratio_w + box[0]
        # y3 = (y + h) * ratio_h + box[1]
        # polygon = [x1, y1, x3, y1, x3, y3, x1, y3]
        regions = seg[1:, temp == 255].reshape((36, -1))
        cs = np.mean(regions, axis=1)
        sym = num2char(np.argmax(cs.reshape((-1))) + 1)
        char["x"] = c_x
        char["y"] = c_y
        char["s"] = sym
        char["cs"] = cs.reshape((-1, 1))
        scores.append(np.max(char["cs"], axis=0)[0])

        chars.append(char)
    chars = sorted(chars, key=lambda x: x["x"])
    string = ""
    css = []
    for char in chars:
        string = string + char["s"]
        css.append(char["cs"])
    if len(scores) > 0:
        score = sum(scores) / len(scores)
    else:
        score = 0.00
    if not css:
        css = [0.0]
    return string, score, np.hstack(css), char_polygons


# def get_tight_rect(points, start_x, start_y, image_height, image_width, scale):
#     points = list(points)
#     ps = sorted(points, key=lambda x: x[0])
#
#     if ps[1][1] > ps[0][1]:
#         px1 = ps[0][0] * scale + start_x
#         py1 = ps[0][1] * scale + start_y
#         px4 = ps[1][0] * scale + start_x
#         py4 = ps[1][1] * scale + start_y
#     else:
#         px1 = ps[1][0] * scale + start_x
#         py1 = ps[1][1] * scale + start_y
#         px4 = ps[0][0] * scale + start_x
#         py4 = ps[0][1] * scale + start_y
#     if ps[3][1] > ps[2][1]:
#         px2 = ps[2][0] * scale + start_x
#         py2 = ps[2][1] * scale + start_y
#         px3 = ps[3][0] * scale + start_x
#         py3 = ps[3][1] * scale + start_y
#     else:
#         px2 = ps[3][0] * scale + start_x
#         py2 = ps[3][1] * scale + start_y
#         px3 = ps[2][0] * scale + start_x
#         py3 = ps[2][1] * scale + start_y
#
#     if px1 < 0:
#         px1 = 1
#     if px1 > image_width:
#         px1 = image_width - 1
#     if px2 < 0:
#         px2 = 1
#     if px2 > image_width:
#         px2 = image_width - 1
#     if px3 < 0:
#         px3 = 1
#     if px3 > image_width:
#         px3 = image_width - 1
#     if px4 < 0:
#         px4 = 1
#     if px4 > image_width:
#         px4 = image_width - 1
#
#     if py1 < 0:
#         py1 = 1
#     if py1 > image_height:
#         py1 = image_height - 1
#     if py2 < 0:
#         py2 = 1
#     if py2 > image_height:
#         py2 = image_height - 1
#     if py3 < 0:
#         py3 = 1
#     if py3 > image_height:
#         py3 = image_height - 1
#     if py4 < 0:
#         py4 = 1
#     if py4 > image_height:
#         py4 = image_height - 1
#     return [px1, py1, px2, py2, px3, py3, px4, py4]

def get_tight_rect(points, start_x, start_y, image_height, image_width, scale):
    points = list(points)
    ps = sorted(points, key=lambda x: x[0])

    if ps[1][1] > ps[0][1]:
        px1 = ps[0][0] * scale + start_x
        py1 = ps[0][1] * scale + start_y
        px4 = ps[1][0] * scale + start_x
        py4 = ps[1][1] * scale + start_y
    else:
        px1 = ps[1][0] * scale + start_x
        py1 = ps[1][1] * scale + start_y
        px4 = ps[0][0] * scale + start_x
        py4 = ps[0][1] * scale + start_y
    if ps[3][1] > ps[2][1]:
        px2 = ps[2][0] * scale + start_x
        py2 = ps[2][1] * scale + start_y
        px3 = ps[3][0] * scale + start_x
        py3 = ps[3][1] * scale + start_y
    else:
        px2 = ps[3][0] * scale + start_x
        py2 = ps[3][1] * scale + start_y
        px3 = ps[2][0] * scale + start_x
        py3 = ps[2][1] * scale + start_y

    px1 = min(max(px1, 1), image_width - 1)
    px2 = min(max(px2, 1), image_width - 1)
    px3 = min(max(px3, 1), image_width - 1)
    px4 = min(max(px4, 1), image_width - 1)
    py1 = min(max(py1, 1), image_height - 1)
    py2 = min(max(py2, 1), image_height - 1)
    py3 = min(max(py3, 1), image_height - 1)
    py4 = min(max(py4, 1), image_height - 1)
    return [px1, py1, px2, py2, px3, py3, px4, py4]
