#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
sys.path.append('./')
import shapely
from shapely.geometry import Polygon,MultiPoint
import numpy as np
import editdistance
sys.path.append('../../')
from weighted_editdistance import weighted_edit_distance
from tqdm import tqdm
try:
    import pickle
except ImportError:
    import cPickle as pickle

def list_from_str(st):
    line = st.split(',')
    # box[0:4], polygon[4:12], word, seq_word, detection_score, rec_socre, seq_score, char_score_path
    new_line = [float(a) for a in line[4:12]]+[float(line[-4])]+[line[-5]]+[line[-6]]+[float(line[-3])]+[float(line[-2])] + [line[-1]]
    return new_line

def polygon_from_list(line):
    """
    Create a shapely polygon object from gt or dt line.
    """
    polygon_points = np.array(line).reshape(4, 2)
    polygon = Polygon(polygon_points).convex_hull
    return polygon

def polygon_iou(list1, list2):
    """
    Intersection over union between two shapely polygons.
    """
    polygon_points1 = np.array(list1).reshape(4, 2)
    poly1 = Polygon(polygon_points1).convex_hull
    polygon_points2 = np.array(list2).reshape(4, 2)
    poly2 = Polygon(polygon_points2).convex_hull
    union_poly = np.concatenate((polygon_points1,polygon_points2))
    if not poly1.intersects(poly2): # this test is fast and can accelerate calculation
        iou = 0
    else:
        try:
            inter_area = poly1.intersection(poly2).area
            #union_area = poly1.area + poly2.area - inter_area
            union_area = MultiPoint(union_poly).convex_hull.area
            iou = float(inter_area) / (union_area+1e-6)
        except shapely.geos.TopologicalError:
            print('shapely.geos.TopologicalError occured, iou set to 0')
            iou = 0
    return iou

def nms(boxes,overlap):
    rec_scores = [b[-2] for b in boxes]
    indices = sorted(range(len(rec_scores)), key=lambda k: -rec_scores[k])
    box_num = len(boxes)
    nms_flag = [True]*box_num
    for i in range(box_num):
        ii = indices[i]
        if not nms_flag[ii]:
            continue
        for j in range(box_num):
            jj = indices[j]
            if j == i:
                continue
            if not nms_flag[jj]:
                continue
            box1 = boxes[ii]
            box2 = boxes[jj] 
            box1_score = rec_scores[ii]
            box2_score = rec_scores[jj] 
            str1 = box1[9] 
            str2 = box2[9]
            box_i = [box1[0],box1[1],box1[4],box1[5]]
            box_j = [box2[0],box2[1],box2[4],box2[5]]
            poly1 = polygon_from_list(box1[0:8])
            poly2 = polygon_from_list(box2[0:8])
            iou = polygon_iou(box1[0:8],box2[0:8])
            thresh = overlap
     
            if iou > thresh:
                if box1_score > box2_score:
                    nms_flag[jj] = False  
                if box1_score == box2_score and poly1.area > poly2.area:
                    nms_flag[jj] = False  
                if box1_score == box2_score and poly1.area<=poly2.area:
                    nms_flag[ii] = False  
                    break

    return nms_flag
         
def packing(save_dir, cache_dir, pack_name):
    files = os.listdir(save_dir)
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)
    os.system('zip -r -q -j '+os.path.join(cache_dir, pack_name+'.zip')+' '+save_dir+'/*')

def test_single(results_dir,lexicon_type=3,cache_dir='./cache_dir',score_det=0.5,score_rec=0.5,score_rec_seq=0.5,overlap=0.2, use_lexicon=True, weighted_ed=True, use_seq=False, use_char=False, mix=False):
    '''
    results_dir: result directory
    score_det: score of detection bounding box
    score_rec: score of the mask recognition branch
    socre_rec_seq: score of the sequence recognition branch
    overlap: overlap threshold used for nms
    lexicon_type: 1 for generic; 2 for weak; 3 for strong
    use_seq: use the recognition result of sequence branch
    use_mix: use both the recognition result of the mask and sequence branches, selected by score
    '''
    print('score_det:', 'score_det:', score_det, 'score_rec:', score_rec, 'score_rec_seq:', score_rec_seq, 'lexicon_type:', lexicon_type, 'weighted_ed:', weighted_ed, 'use_seq:', use_seq, 'use_char:', use_char, 'mix:', mix)
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)   
    nms_dir = os.path.join(cache_dir,str(score_det)+'_'+str(score_rec)+'_'+str(score_rec_seq))    
    if not os.path.exists(nms_dir):
        os.mkdir(nms_dir)
    if lexicon_type==1: 
        # generic lexicon
        lexicon_path = '../../lexicons/ic13/GenericVocabulary_new.txt'
        lexicon_fid=open(lexicon_path, 'r')
        pair_list = open('../../lexicons/ic13/GenericVocabulary_pair_list.txt', 'r')
        pairs = dict()
        for line in pair_list.readlines():
            line=line.strip()
            word = line.split(' ')[0].upper()
            word_gt = line[len(word)+1:]
            pairs[word] = word_gt
        lexicon_fid=open(lexicon_path, 'r')
        lexicon=[]
        for line in lexicon_fid.readlines():
            line=line.strip()
            lexicon.append(line)
    if lexicon_type==2:
        # weak lexicon
        lexicon_path = '../../lexicons/ic13/ch4_test_vocabulary_new.txt'
        lexicon_fid=open(lexicon_path, 'r')
        pair_list = open('../../lexicons/ic13/ch4_test_vocabulary_pair_list.txt', 'r')
        pairs = dict()
        for line in pair_list.readlines():
            line=line.strip()
            word = line.split(' ')[0].upper()
            word_gt = line[len(word)+1:]
            pairs[word] = word_gt
        lexicon_fid=open(lexicon_path, 'r')
        lexicon=[]
        for line in lexicon_fid.readlines():
            line=line.strip()
            lexicon.append(line)

    for i in tqdm(range(1,234)):
        img = 'img_'+str(i)+'.jpg'
        gt_img = 'gt_img_'+str(i)+'.txt'
        if lexicon_type==3:
            # weak
            lexicon_path = '../../lexicons/ic13/new_strong_lexicon/new_voc_img_' + str(i) + '.txt'
            lexicon_fid=open(lexicon_path, 'r')
            pair_list = open('../../lexicons/ic13/new_strong_lexicon/pair_voc_img_' + str(i) + '.txt', 'r')
            pairs = dict()
            for line in pair_list.readlines():
                line=line.strip()
                word = line.split(' ')[0].upper()
                word_gt = line[len(word)+1:]
                pairs[word] = word_gt
            lexicon_fid=open(lexicon_path, 'r')
            lexicon=[]
            for line in lexicon_fid.readlines():
                line=line.strip()
                lexicon.append(line)
        result_path = os.path.join(results_dir,'res_img_'+str(i)+'.txt')
        if os.path.isfile(result_path):
            with open(result_path,'r') as f:
                dt_lines = [a.strip() for a in f.readlines()]
            dt_lines = [list_from_str(dt) for dt in dt_lines]
        else:
            dt_lines = []
        dt_lines = [dt for dt in dt_lines if dt[-2]>score_rec_seq and dt[-3]>score_rec and dt[-6]>score_det]
        nms_flag = nms(dt_lines,overlap)
        boxes = []
        for k in range(len(dt_lines)):
            dt = dt_lines[k]
            if nms_flag[k]:
                if dt not in boxes:
                    boxes.append(dt)
        
        with open(os.path.join(nms_dir,'res_img_'+str(i)+'.txt'),'w') as f:
            for g in boxes:
                gt_coors = [int(b) for b in g[0:8]]
                with open('../../../' + g[-1], "rb") as input_file:
                # with open(g[-1], "rb") as input_file:
                    dict_scores = pickle.load(input_file)
                if use_char and use_seq:
                    if g[-2]>g[-3]:
                        word = g[-5]
                        scores = dict_scores['seq_char_scores'][:,1:-1].swapaxes(0,1)
                    else:
                        word = g[-4]
                        scores = dict_scores['seg_char_scores']
                elif use_seq:
                    word = g[-5]
                    scores = dict_scores['seq_char_scores'][:,1:-1].swapaxes(0,1)
                else:
                    word = g[-4]
                    scores = dict_scores['seg_char_scores']
                if not use_lexicon:
                    match_word = word
                    match_dist = 0.
                else:
                    match_word, match_dist = find_match_word(word, lexicon, pairs, scores, use_lexicon, weighted_ed)
                if match_dist<1.5 or lexicon_type==1:
                    gt_coor_strs = [str(a) for a in gt_coors]+ [match_word]
                    f.write(','.join(gt_coor_strs)+'\r\n')

    pack_name = str(score_det)+'_'+str(score_rec)+'_over'+str(overlap)       

    packing(nms_dir,cache_dir,pack_name)
    submit_file_path = os.path.join(cache_dir, pack_name+'.zip')
    return submit_file_path
   
def find_match_word(rec_str, lexicon, pairs, scores_numpy, use_ed = True, weighted_ed = False):
    if not use_ed:
        return rec_str
    rec_str = rec_str.upper()
    dist_min = 100
    dist_min_pre = 100
    match_word = ''
    match_dist = 100
    if not weighted_ed:
        for word in lexicon:
            word = word.upper()
            ed = editdistance.eval(rec_str, word)
            length_dist = abs(len(word) - len(rec_str))
            # dist = ed + length_dist
            dist = ed
            if dist<dist_min:
                dist_min = dist
                match_word = pairs[word]
                match_dist = dist
        return match_word, match_dist
    else:
        small_lexicon_dict = dict()
        for word in lexicon:
            word = word.upper()
            ed = editdistance.eval(rec_str, word)
            small_lexicon_dict[word] = ed
            dist = ed
            if dist<dist_min_pre:
                dist_min_pre = dist
        small_lexicon = []
        for word in small_lexicon_dict:
            if small_lexicon_dict[word]<=dist_min_pre+2:
                small_lexicon.append(word)

        for word in small_lexicon:
            word = word.upper()
            ed = weighted_edit_distance(rec_str, word, scores_numpy)
            dist = ed
            if dist<dist_min:
                dist_min = dist
                match_word = pairs[word]
                match_dist = dist
        return match_word, match_dist


def prepare_results_for_evaluation(results_dir, use_lexicon, cache_dir, score_det, score_rec, score_rec_seq):
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    result_path = test_single(results_dir,score_det=score_det,score_rec=score_rec,score_rec_seq=score_rec_seq,overlap=0.2,cache_dir=cache_dir,lexicon_type=3, use_lexicon=use_lexicon, weighted_ed=True, use_seq=True, use_char=True, mix=True)
    return result_path