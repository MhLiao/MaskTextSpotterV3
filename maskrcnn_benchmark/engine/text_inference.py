# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import os
import pickle
import subprocess
import time

import cv2
import numpy as np
import torch
from maskrcnn_benchmark.utils.chars import char2num, get_tight_rect, getstr_grid
from PIL import Image, ImageDraw
from tqdm import tqdm

from ..utils.comm import is_main_process, scatter_gather, synchronize
import pdb

# TO DO: format output with dictionnary
def compute_on_dataset(model, data_loader, device, cfg):
    model.eval()
    results_dict = {}
    seg_results = []
    cpu_device = torch.device("cpu")
    total_time = 0
    for _, batch in tqdm(enumerate(data_loader)):
        images, targets, image_paths = batch
        images = images.to(device)
        with torch.no_grad():
            if cfg.MODEL.SEG_ON:
                predictions, proposals, seg_results_dict = model(
                    images
                )
                seg_results.append(
                    [image_paths, proposals, seg_results_dict['rotated_boxes'], seg_results_dict['polygons'], seg_results_dict['preds'], seg_results_dict['scores']]
                ) 
                # if cfg.MODEL.MASK_ON and predictions is not None:
                if predictions is not None:
                    if cfg.MODEL.CHAR_MASK_ON or cfg.SEQUENCE.SEQ_ON:
                        global_predictions = predictions[0]
                        char_predictions = predictions[1]
                        char_mask = char_predictions["char_mask"]
                        boxes = char_predictions["boxes"]
                        seq_words = char_predictions["seq_outputs"]
                        seq_scores = char_predictions["seq_scores"]
                        detailed_seq_scores = char_predictions["detailed_seq_scores"]
                        global_predictions = [o.to(cpu_device) for o in global_predictions]
                        results_dict.update(
                            {
                                image_paths[0]: [
                                    global_predictions[0],
                                    char_mask,
                                    boxes,
                                    seq_words,
                                    seq_scores,
                                    detailed_seq_scores,
                                ]
                            }
                        )
                    else:
                        global_predictions = [o.to(cpu_device) for o in predictions]
                        results_dict.update(
                            {
                                image_paths[0]: [
                                    global_predictions[0],
                                ]
                            }
                        )
            else:
                predictions = model(images)
                if predictions is not None:
                    if not (cfg.MODEL.CHAR_MASK_ON and cfg.SEQUENCE.SEQ_ON):
                        global_predictions = predictions
                        global_predictions = [o.to(cpu_device) for o in global_predictions]
                        results_dict.update(
                            {
                                image_paths[0]: [
                                    global_predictions[0],
                                ]
                            }
                        )
                    else:
                        global_predictions = predictions[0]
                        char_predictions = predictions[1]
                        if cfg.MODEL.CHAR_MASK_ON:
                            char_mask = char_predictions["char_mask"]
                        else:
                            char_mask = None
                        boxes = char_predictions["boxes"]
                        seq_words = char_predictions["seq_outputs"]
                        seq_scores = char_predictions["seq_scores"]
                        detailed_seq_scores = char_predictions["detailed_seq_scores"]
                        global_predictions = [o.to(cpu_device) for o in global_predictions]
                        results_dict.update(
                            {
                                image_paths[0]: [
                                    global_predictions[0],
                                    char_mask,
                                    boxes,
                                    seq_words,
                                    seq_scores,
                                    detailed_seq_scores,
                                ]
                            }
                        )
    return results_dict, seg_results


def polygon2rbox(polygon, image_height, image_width):
    poly = np.array(polygon).reshape((-1, 2))
    rect = cv2.minAreaRect(poly)
    corners = cv2.boxPoints(rect)
    corners = np.array(corners, dtype="int")
    pts = get_tight_rect(corners, 0, 0, image_height, image_width, 1)
    pts = list(map(int, pts))
    return pts


def mask2polygon(mask, box, im_size, threshold=0.5, output_folder=None):
    # mask 32*128
    image_width, image_height = im_size[0], im_size[1]
    box_h = box[3] - box[1]
    box_w = box[2] - box[0]
    cls_polys = (mask * 255).astype(np.uint8)
    poly_map = np.array(Image.fromarray(cls_polys).resize((box_w, box_h)))
    poly_map = poly_map.astype(np.float32) / 255
    poly_map = cv2.GaussianBlur(poly_map, (3, 3), sigmaX=3)
    ret, poly_map = cv2.threshold(poly_map, threshold, 1, cv2.THRESH_BINARY)
    if "total_text" in output_folder or "cute80" in output_folder:
        SE1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        poly_map = cv2.erode(poly_map, SE1)
        poly_map = cv2.dilate(poly_map, SE1)
        poly_map = cv2.morphologyEx(poly_map, cv2.MORPH_CLOSE, SE1)
        try:
            _, contours, _ = cv2.findContours(
                (poly_map * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
            )
        except:
            contours, _ = cv2.findContours(
                (poly_map * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
            )
        if len(contours) == 0:
            # print(contours)
            # print(len(contours))
            return None
        max_area = 0
        max_cnt = contours[0]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                max_cnt = cnt
        # perimeter = cv2.arcLength(max_cnt, True)
        epsilon = 0.01 * cv2.arcLength(max_cnt, True)
        approx = cv2.approxPolyDP(max_cnt, epsilon, True)
        pts = approx.reshape((-1, 2))
        pts[:, 0] = pts[:, 0] + box[0]
        pts[:, 1] = pts[:, 1] + box[1]
        polygon = list(pts.reshape((-1,)))
        polygon = list(map(int, polygon))
        if len(polygon) < 6:
            return None
    else:
        SE1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        poly_map = cv2.erode(poly_map, SE1)
        poly_map = cv2.dilate(poly_map, SE1)
        poly_map = cv2.morphologyEx(poly_map, cv2.MORPH_CLOSE, SE1)
        idy, idx = np.where(poly_map == 1)
        xy = np.vstack((idx, idy))
        xy = np.transpose(xy)
        hull = cv2.convexHull(xy, clockwise=True)
        # reverse order of points.
        if hull is None:
            return None
        hull = hull[::-1]
        # find minimum area bounding box.
        rect = cv2.minAreaRect(hull)
        corners = cv2.boxPoints(rect)
        corners = np.array(corners, dtype="int")
        pts = get_tight_rect(corners, box[0], box[1], image_height, image_width, 1)
        polygon = [x * 1.0 for x in pts]
        polygon = list(map(int, polygon))
    return polygon


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = scatter_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    return predictions


def format_output(out_dir, boxes, img_name):
    with open(
        os.path.join(out_dir, "res_" + img_name.split(".")[0] + ".txt"), "wt"
    ) as res:
        ## char score save dir
        ssur_name = os.path.join(out_dir, "res_" + img_name.split(".")[0])
        for i, box in enumerate(boxes):
            save_name = ssur_name + "_" + str(i) + ".pkl"
            save_dict = {}
            if "total_text" in out_dir or "cute80" in out_dir:
                # np.save(save_name, box[-2])
                save_dict["seg_char_scores"] = box[-3]
                save_dict["seq_char_scores"] = box[-2]
                box = (
                    ",".join([str(x) for x in box[:4]])
                    + ";"
                    + ",".join([str(x) for x in box[4 : 4 + int(box[-1])]])
                    + ";"
                    + ",".join([str(x) for x in box[4 + int(box[-1]) : -3]])
                    + ","
                    + save_name
                )
            else:
                save_dict["seg_char_scores"] = box[-2]
                save_dict["seq_char_scores"] = box[-1]
                np.save(save_name, box[-1])
                box = ",".join([str(x) for x in box[:-2]]) + "," + save_name
            with open(save_name, "wb") as f:
                pickle.dump(save_dict, f, protocol=2)
            res.write(box + "\n")

def format_seg_output(results_dir, rotated_boxes_this_image, polygons_this_image, scores, img_name, ratio):
    height_ratio, width_ratio = ratio
    with open(
        os.path.join(results_dir, "res_" + img_name.split(".")[0] + ".txt"), "wt"
    ) as res:
        if "total_text" in results_dir or "cute80" in results_dir:
            for i, box in enumerate(polygons_this_image):
                box = box[0]
                box[0::2] = box[0::2] * width_ratio
                box[1::2] = box[1::2] * height_ratio
                save_dict = {}
                # result = ",".join([str(int(x[0])) + ',' +str(int(x[1])) for x in box])
                result = ",".join([str(int(x)) for x in box])
                score = scores[i].item()
                res.write(result + ',' + str(score) + "\n")
        else:
            for i, box in enumerate(rotated_boxes_this_image):
                box[0::2] = box[0::2] * width_ratio
                box[1::2] = box[1::2] * height_ratio
                save_dict = {}
                result = ",".join([str(int(x[0])) + ',' +str(int(x[1])) for x in box])
                score = scores[i].item()
                res.write(result + ',' + str(score) + "\n")



def process_char_mask(char_masks, boxes, threshold=192):
    texts, rec_scores, rec_char_scores, char_polygons = [], [], [], []
    for index in range(char_masks.shape[0]):
        box = list(boxes[index])
        box = list(map(int, box))
        text, rec_score, rec_char_score, char_polygon = getstr_grid(
            char_masks[index, :, :, :].copy(), box, threshold=threshold
        )
        texts.append(text)
        rec_scores.append(rec_score)
        rec_char_scores.append(rec_char_score)
        char_polygons.append(char_polygon)
        # segmss.append(segms)
    return texts, rec_scores, rec_char_scores, char_polygons


def creat_color_map(n_class, width):
    splits = int(np.ceil(np.power((n_class * 1.0), 1.0 / 3)))
    maps = []
    for i in range(splits):
        r = int(i * width * 1.0 / (splits - 1))
        for j in range(splits):
            g = int(j * width * 1.0 / (splits - 1))
            for k in range(splits - 1):
                b = int(k * width * 1.0 / (splits - 1))
                maps.append((r, g, b, 200))
    return maps


def visualization(image, polygons, resize_ratio, colors, char_polygons=None, words=None):
    draw = ImageDraw.Draw(image, "RGBA")
    for polygon in polygons:
        # draw.polygon(polygon, fill=None, outline=(0, 255, 0, 255))
        # print(polygon)
        polygon.append(polygon[0])
        polygon.append(polygon[1])
        # print(polygon)
        color = '#33FF33'
        draw.line(polygon, fill=color, width=5)
    # if char_polygons is not None:
    #     for i, char_polygon in enumerate(char_polygons):
    #         for j, polygon in enumerate(char_polygon):
    #             polygon = [int(x * resize_ratio) for x in polygon]
    #             char = words[i][j]
    #             color = colors[char2num(char)]
    #             draw.polygon(polygon, fill=color, outline=color)


def vis_seg_map(image_path, seg_map, rotated_boxes, polygons_this_image, proposals, vis_dir):
    img_name = image_path.split("/")[-1]
    image = cv2.imread(image_path)
    height, width, _ = image.shape
    seg_map = seg_map.data.cpu().numpy()
    img = Image.fromarray(image).convert("RGB")
    # height_ratio = height / seg_map.shape[1]
    # width_ratio = width / seg_map.shape[2]
    # print('seg_map.shape:', seg_map.shape)
    # print('image.shape:', image.shape)
    seg_image = (
        Image.fromarray((seg_map[0, :proposals.size[1], :proposals.size[0]] * 255).astype(np.uint8))
        .convert("RGB")
        .resize((width, height))
    )
    visu_image = Image.blend(seg_image, img, 0.5)
    img_draw = ImageDraw.Draw(visu_image)
    if "total_text" in vis_dir or "cute80" in vis_dir:
        for box in polygons_this_image:
            # box[:, 0] = box[:, 0]
            # box[:, 1] = box[:, 1]
            tuple_box = [tuple(x) for x in box[0].reshape(-1, 2).tolist()]
            tuple_box.append(tuple_box[0])
            img_draw.line(tuple_box, fill=(0, 255, 0), width=5)
    else:
        for box in rotated_boxes:
            # box[:, 0] = box[:, 0]
            # box[:, 1] = box[:, 1]
            tuple_box = [tuple(x) for x in box.tolist()]
            tuple_box.append(tuple_box[0])
            img_draw.line(tuple_box, fill=(0, 255, 0), width=5)
    visu_image.save(vis_dir + "/seg_" + img_name)


def prepare_results_for_evaluation(
    predictions, output_folder, model_name, seg_predictions=None, vis=False, cfg=None
):
    results_dir = os.path.join(output_folder, model_name + "_results")
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    seg_results_dir = os.path.join(output_folder, model_name + "_seg_results")
    if not os.path.isdir(seg_results_dir):
        os.mkdir(seg_results_dir)
    if vis:
        visu_dir = os.path.join(output_folder, model_name + "_visu")
        if not os.path.isdir(visu_dir):
            os.mkdir(visu_dir)
        seg_visu_dir = os.path.join(output_folder, model_name + "_seg_visu")
        if not os.path.isdir(seg_visu_dir):
            os.mkdir(seg_visu_dir)
    if len(seg_predictions) > 0:
        for seg_prediction in seg_predictions:
            image_paths, proposals, rotated_boxes, polygons, seg_maps, seg_scores = (
                seg_prediction[0],
                seg_prediction[1],
                seg_prediction[2],
                seg_prediction[3],
                seg_prediction[4],
                seg_prediction[5],
            )
            for batch_id in range(len(image_paths)):
                image_path = image_paths[batch_id]
                im_name = image_path.split("/")[-1]
                image = cv2.imread(image_path)
                height, width, _ = image.shape
                rotated_boxes_this_image = rotated_boxes[batch_id]
                polygons_this_image = polygons[batch_id]
                proposals_this_image = proposals[batch_id]
                seg_map = seg_maps[batch_id]
                seg_score = seg_scores[batch_id]
                height, width, _ = image.shape
                height_ratio = height / proposals_this_image.size[1]
                width_ratio = width / proposals_this_image.size[0]
                format_seg_output(seg_results_dir, rotated_boxes_this_image, polygons_this_image, seg_score, im_name, (height_ratio, width_ratio))
                if vis:
                    vis_seg_map(image_path, seg_map, rotated_boxes_this_image, polygons_this_image, proposals_this_image, seg_visu_dir)
    if (not cfg.MODEL.TRAIN_DETECTION_ONLY):
        for image_path, prediction in predictions.items():
            im_name = image_path.split("/")[-1]
            if cfg.MODEL.CHAR_MASK_ON or cfg.SEQUENCE.SEQ_ON:
                global_prediction, char_mask, boxes_char, seq_words, seq_scores, detailed_seq_scores = (
                    prediction[0],
                    prediction[1],
                    prediction[2],
                    prediction[3],
                    prediction[4],
                    prediction[5],
                )
                if char_mask is not None:
                    words, rec_scores, rec_char_scoress, char_polygons = process_char_mask(
                        char_mask, boxes_char
                    )
            else:
                global_prediction = prediction[0]
            test_image_width, test_image_height = global_prediction.size
            img = Image.open(image_path)
            width, height = img.size
            resize_ratio = float(height) / test_image_height
            global_prediction = global_prediction.resize((width, height))
            boxes = global_prediction.bbox.tolist()
            if cfg.MODEL.ROI_BOX_HEAD.INFERENCE_USE_BOX:
                scores = global_prediction.get_field("scores").tolist()
            if not cfg.MODEL.SEG.USE_SEG_POLY:
                masks = global_prediction.get_field("mask").cpu().numpy()
            else:
                masks = global_prediction.get_field("masks").get_polygons()
            result_logs = []
            polygons = []
            for k, box in enumerate(boxes):
                if box[2] - box[0] < 1 or box[3] - box[1] < 1:
                    continue
                box = list(map(int, box))
                if not cfg.MODEL.SEG.USE_SEG_POLY:
                    mask = masks[k, 0, :, :]
                    polygon = mask2polygon(
                        mask, box, img.size, threshold=0.5, output_folder=output_folder
                    )
                else:
                    polygon = list(masks[k].get_polygons()[0].cpu().numpy())
                    if not ("total_text" in output_folder or "cute80" in output_folder):
                        polygon = polygon2rbox(polygon, height, width)
                if polygon is None:
                    polygon = [
                        box[0],
                        box[1],
                        box[2],
                        box[1],
                        box[2],
                        box[3],
                        box[0],
                        box[3],
                    ]
                    continue
                polygons.append(polygon)
                if cfg.MODEL.ROI_BOX_HEAD.INFERENCE_USE_BOX:
                    score = scores[k]
                else:
                    score = 1.0
                if cfg.MODEL.CHAR_MASK_ON or cfg.SEQUENCE.SEQ_ON:
                    if char_mask is None:
                        word = 'aaa'
                        rec_score = 1.0
                        char_score = None
                    else:
                        word = words[k]
                        rec_score = rec_scores[k]
                        char_score = rec_char_scoress[k]
                    seq_word = seq_words[k]
                    seq_char_scores = seq_scores[k]
                    seq_score = sum(seq_char_scores) / float(len(seq_char_scores))
                    detailed_seq_score = detailed_seq_scores[k]
                    detailed_seq_score = np.squeeze(np.array(detailed_seq_score), axis=1)
                else:
                    word = 'aaa'
                    rec_score = 1.0
                    char_score = [1.0, 1.0, 1.0]
                    seq_word = 'aaa'
                    seq_char_scores = [1.0, 1.0, 1.0]
                    seq_score = 1.0
                    detailed_seq_score = None
                if "total_text" in output_folder or "cute80" in output_folder:
                    result_log = (
                        [int(x * 1.0) for x in box[:4]]
                        + polygon
                        + [word]
                        + [seq_word]
                        + [score]
                        + [rec_score]
                        + [seq_score]
                        + [char_score]
                        + [detailed_seq_score]
                        + [len(polygon)]
                    )
                else:
                    result_log = (
                        [int(x * 1.0) for x in box[:4]]
                        + polygon
                        + [word]
                        + [seq_word]
                        + [score]
                        + [rec_score]
                        + [seq_score]
                        + [char_score]
                        + [detailed_seq_score]
                    )
                result_logs.append(result_log)
            if vis:
                colors = creat_color_map(37, 255)
                if cfg.MODEL.CHAR_MASK_ON:
                    visualization(img, polygons, resize_ratio, colors, char_polygons, words)
                else:
                    visualization(img, polygons, resize_ratio, colors)
                img.save(os.path.join(visu_dir, im_name))
            format_output(results_dir, result_logs, im_name)


def inference(
    model,
    data_loader,
    iou_types=("bbox",),
    box_only=False,
    device="cuda",
    expected_results=(),
    expected_results_sigma_tol=4,
    output_folder=None,
    model_name=None,
    cfg=None,
):

    # convert to a torch.device for efficiency
    model_name = model_name.split(".")[0] + "_" + str(cfg.INPUT.MIN_SIZE_TEST)
    predictions_path = os.path.join(output_folder, model_name + "_predictions.pth")
    seg_predictions_path = os.path.join(
        output_folder, model_name + "_seg_predictions.pth"
    )
    # if os.path.isfile(predictions_path) and os.path.isfile(seg_predictions_path):
    if False:
        predictions = torch.load(predictions_path)
        seg_predictions = torch.load(seg_predictions_path)
    else:
        device = torch.device(device)
        num_devices = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        dataset = data_loader.dataset
        logger.info("Start evaluation on {} images".format(len(dataset)))
        start_time = time.time()
        predictions, seg_predictions = compute_on_dataset(
            model, data_loader, device, cfg
        )
        # wait for all processes to complete before measuring the time
        synchronize()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        logger.info(
            "Total inference time: {} ({} s / img per device, on {} devices)".format(
                total_time_str, total_time * num_devices / len(dataset), num_devices
            )
        )

        # predictions = _accumulate_predictions_from_multiple_gpus(predictions)
        # if not is_main_process():
        # 	return

        if output_folder:
            torch.save(predictions, predictions_path)
            torch.save(seg_predictions, seg_predictions_path)

    prepare_results_for_evaluation(
        predictions,
        output_folder,
        model_name,
        seg_predictions=seg_predictions,
        vis=cfg.TEST.VIS,
        cfg=cfg
    )
