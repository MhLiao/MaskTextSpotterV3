'''
Author: Mohammed Innat
mail: innat1994@gmail.com
'''

import shapely
import numpy as np, glob, cv2
import matplotlib.pyplot as plt 
from shapely.geometry import Polygon 

# find iou over two polygons
def polygon_iou(poly1, poly2):
    if not poly1.intersects(poly2): 
        iou = 0
    else:
        try:
            pol_intersection = poly1.intersection(poly2).area
            pol_union        = poly1.union(poly2).area
            iou = pol_intersection/pol_union
        except (shapely.geos.TopologicalError, ZeroDivisionError):
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou

# non-max supression over polygons 
def poly_nms(dets, thresh):
    scores = dets[:, -2]
    order  = scores.argsort()[::-1]
    polys  = []
    
    for i in range(len(dets)):
        poly_pts = np.array(dets[i, :-2][0].reshape(-1,2))
        polygon  = Polygon(poly_pts).convex_hull
        polys.append(polygon)
        
    keep_idx = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep_idx.append(i)
        
        for j in range(order.size - 1):
            iou = polygon_iou(polys[i], polys[order[j + 1]])
            ovr.append(iou)
        ovr   = np.array(ovr)
        inds  = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep_idx

def get_cords(label, data, xx, yy, cts, scores):
    '''
    input:
        - label: a text format annotation of polygons in: x,y,x,y...transcription
        - data , xx , yy, cts <- an empty list 
        
    return:
        - data: contains each line from the text file corresponding to one transcription 
        - xx  : all x cords
        - yy  : all y cords
        - cts : all transcriptions
    '''
    fin = open(label, 'r').readlines()
    for il, line in enumerate(fin):
        line = line.strip().split(',')
        ct = line[-2]
        score = line[-1]
        for ix in range(0, len(line[:-2]), 2):
            xx.append(float(line[:-2][ix]))
            yy.append(float(line[:-2][ix+1]))
        cts.append(ct)
        scores.append(score)
        data.append(np.array([float(x) for x in line[:-2]]))
    return data, xx, yy, cts, scores


def vis_polygons(image, polygons):
    '''
    It will plot polygons with connected keypoints (poly-lines) with transcriptions
    '''
    for polygon in polygons:
        #print(polygon)
        pts = np.array(polygon, np.int32)
        pts = pts.reshape((-1,1,2))
        xmin = min(pts[:,0,0])
        ymin = min(pts[:,0,1])
        cv2.polylines(image,[pts],True,(0,255,0), 2)

