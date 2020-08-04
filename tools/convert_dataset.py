import os
import numpy as np
import cv2
from shapely.geometry import box, Polygon 
from shapely import affinity
import math


def _rect2quad(boxes):
    x_min, y_min, x_max, y_max = boxes[:, 0].reshape((-1, 1)), boxes[:, 1].reshape((-1, 1)), boxes[:, 2].reshape((-1, 1)), boxes[:, 3].reshape((-1, 1))
    return np.hstack((x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max))

def _quad2rect(boxes):
    ## only support rectangle
    return np.hstack((boxes[:, 0].reshape((-1, 1)), boxes[:, 1].reshape((-1, 1)), boxes[:, 4].reshape((-1, 1)), boxes[:, 5].reshape((-1, 1))))

def _quad2minrect(boxes):
    ## trans a quad(N*4) to a rectangle(N*4) which has miniual area to cover it
    return np.hstack((boxes[:, ::2].min(axis=1).reshape((-1, 1)), boxes[:, 1::2].min(axis=1).reshape((-1, 1)), boxes[:, ::2].max(axis=1).reshape((-1, 1)), boxes[:, 1::2].max(axis=1).reshape((-1, 1))))


def _quad2boxlist(boxes):
    res = []
    for i in range(boxes.shape[0]):
        res.append([[boxes[i][0], boxes[i][1]], [boxes[i][2], boxes[i][3]], [boxes[i][4], boxes[i][5]], [boxes[i][6], boxes[i][7]]])
    return res

def _boxlist2quads(boxlist):
    res = np.zeros((len(boxlist), 8))
    for i, box in enumerate(boxlist):
        # print(box)
        res[i] = np.array([box[0][0], box[0][1], box[1][0], box[1][1], box[2][0], box[2][1], box[3][0], box[3][1]])
    return res

def _rotate_image(im, polygons, angle):
    new_polygons = polygons
    ## rotate image first
    height, width, _ = im.shape
    ## get the minimal rect to cover the rotated image
    img_box = np.array([[0, 0, width, 0, width, height, 0, height]])
    rotated_img_box = _quad2minrect(_rotate_polygons(img_box, -1*angle, (width/2, height/2)))
    r_height = int(max(rotated_img_box[0][3], rotated_img_box[0][1]) - min(rotated_img_box[0][3], rotated_img_box[0][1]))
    r_width = int(max(rotated_img_box[0][2], rotated_img_box[0][0]) - min(rotated_img_box[0][2], rotated_img_box[0][0]))
    r_height_padding = max(r_height, height)
    r_width_padding = max(r_width, width)
    ## padding im
    im_padding = np.zeros((r_height_padding, r_width_padding, 3))
    start_h, start_w = int((r_height_padding - height)/2.0), int((r_width_padding - width)/2.0)
    # start_h = max(start_h, 0)
    # start_w = max(start_w, 0)
    end_h, end_w = start_h + height, start_w + width
    # print(start_h, end_h, start_w, end_w, im.shape)
    im_padding[start_h:end_h, start_w:end_w, :] = im

    M = cv2.getRotationMatrix2D((r_width/2, r_height/2), angle, 1)
    im = cv2.warpAffine(im_padding, M, (r_width, r_height))
    
    ## polygons
    new_polygons = _rotate_segms(polygons, -1*angle, (r_width/2, r_height/2), start_h, start_w)

    return im, new_polygons

def _rotate_polygons(polygons, angle, r_c):
    ## polygons: N*8
    ## r_x: rotate center x
    ## r_y: rotate center y
    ## angle: -15~15

    poly_list = _quad2boxlist(polygons)
    rotate_boxes_list = []
    for poly in poly_list:
        box = Polygon(poly)
        rbox = affinity.rotate(box, angle, r_c)
        if len(list(rbox.exterior.coords))<5:
            print(poly)
            print(rbox)
        # assert(len(list(rbox.exterior.coords))>=5)
        rotate_boxes_list.append(rbox.boundary.coords[:-1])
    res = _boxlist2quads(rotate_boxes_list)
    return res

def _rotate_segms(polygons, angle, r_c, start_h, start_w):
    ## polygons: N*8
    ## r_x: rotate center x
    ## r_y: rotate center y
    ## angle: -15~15
    poly_list=[]
    for polygon in polygons:
        tmp=[]
        for i in range(int(len(polygon) / 2)):
            tmp.append([polygon[2*i] + start_w, polygon[2*i+1] + start_h])
        poly_list.append(tmp)

    rotate_boxes_list = []
    for poly in poly_list:
        box = Polygon(poly)
        rbox = affinity.rotate(box, angle, r_c)
        if len(list(rbox.exterior.coords))<5:
            print(poly)
            print(rbox)
        rotate_boxes_list.append(rbox.boundary.coords[:-1])
    res = []
    for i, box in enumerate(rotate_boxes_list):
        tmp = []
        for point in box:
            tmp.append(point[0])
            tmp.append(point[1])
        res.append([tmp])

    return res

def _read_gt(gt_path):
    polygons = []
    words = []
    with open(gt_path, 'r') as fid:
        lines = fid.readlines()
        for line in lines:
            line = line.strip()
            polygon = line.split(',')[:8]
            word = line.split(',')[8]
            polygon = [float(x) for x in polygon]
            polygons.append(polygon)
            words.append(word)
    return polygons, words

def format_new_gt(polygons, words, new_gt_path):
    with open(new_gt_path, 'wt') as fid:
        for polygon, word in zip(polygons, words):
            # print(polygon)
            polygon = [str(int(x)) for x in polygon[0]]
            # polygon = [str(int(x)) for x in polygon]
            line = ','.join(polygon) + ',' + word
            # print(line)
            fid.write(line+'\n')

def visu_gt(img, polygons, visu_path):
    for polygon in polygons:
        pts = np.array(polygon, np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img,[pts],True,(0,255,255))
    cv2.imwrite(visu_path, img)


img_dir = '../datasets/icdar2013/test_images'
gt_dir = '../datasets/icdar2013/test_gts'
angle = 45
new_img_dir = '../datasets/icdar2013/rotated_test_images'+'_'+str(angle)
new_gt_dir = '../datasets/icdar2013/rotated_test_gts'+'_'+str(angle)
if not os.path.isdir(new_img_dir):
    os.mkdir(new_img_dir)
if not os.path.isdir(new_gt_dir):
    os.mkdir(new_gt_dir)

visu_dir = '../output/visu/'

for i in range(233):
    img_name = 'img_' + str(i+1) + '.jpg'
    img_path = os.path.join(img_dir, img_name)
    img = cv2.imread(img_path)
    gt_path = os.path.join(gt_dir, img_name + '.txt')
    new_img_path = os.path.join(new_img_dir, img_name)
    visu_path = os.path.join(visu_dir, img_name)
    new_gt_path = os.path.join(new_gt_dir, 'gt_' + img_name.split('.')[0] + '.txt')
    polygons, words = _read_gt(gt_path)
    # print(img_name)
    if angle == 90:
        (h, w) = img.shape[:2]
        img = cv2.transpose(img)
        img = cv2.flip(img,flipCode=0)
        # M = cv2.getRotationMatrix2D(center, 90, 1)
        # img = cv2.warpAffine(img, M, (h, w))
        new_polygons = [[polygon[1], w-polygon[0], polygon[3], w-polygon[2], polygon[5], w-polygon[4], polygon[7], w-polygon[6]] for polygon in polygons]
    else:
        img, new_polygons = _rotate_image(img, polygons, angle)
    format_new_gt(new_polygons, words, new_gt_path)
    # visu_gt(img, new_polygons, visu_path)
    cv2.imwrite(new_img_path, img)
    