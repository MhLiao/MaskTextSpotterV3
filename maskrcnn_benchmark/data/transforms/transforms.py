# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random

import cv2
import numpy as np
from PIL import Image
from shapely import affinity
from shapely.geometry import Polygon
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class Resize(object):
    def __init__(self, min_size, max_size, strict_resize):
        self.min_size = min_size
        self.max_size = max_size
        self.strict_resize = strict_resize

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        if isinstance(self.min_size, tuple):
            if len(self.min_size) == 1:
                size = self.min_size[0]
            else:
                random_size_index = random.randint(0, len(self.min_size) - 1)
                size = self.min_size[random_size_index]
        else:
            size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            if self.strict_resize:
                h = h if h % 32 == 0 else (h // 32) * 32
                w = w if w % 32 == 0 else (w // 32) * 32
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        if self.strict_resize:
            oh = oh if oh % 32 == 0 else (oh // 32) * 32
            ow = ow if ow % 32 == 0 else (ow // 32) * 32

        return (oh, ow)

    def __call__(self, image, target):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        if target is not None:
            target = target.resize(image.size)
        return image, target


class RandomCrop(object):
    def __init__(self, prob, crop_min_size=500, crop_max_size=1000, max_trys=50):
        self.min_size = crop_min_size
        self.max_size = crop_max_size
        self.max_trys = max_trys
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            im = np.array(image)
            w, h = image.size
            h_array = np.zeros((h), dtype=np.int32)
            w_array = np.zeros((w), dtype=np.int32)
            boxes = target.bbox.numpy()
            if len(boxes) == 0:
                return image, target
            for box in boxes:
                box = np.round(box, decimals=0).astype(np.int32)
                minx = box[0]
                maxx = box[2]
                w_array[minx:maxx] = 1
                miny = box[1]
                maxy = box[3]
                h_array[miny:maxy] = 1
            h_axis = np.where(h_array == 0)[0]
            w_axis = np.where(w_array == 0)[0]
            if len(h_axis) == 0 or len(w_axis) == 0:
                return image, target
            for _ in range(self.max_trys):
                xx = np.random.choice(w_axis, size=2)
                xmin = min(xx)
                xmax = max(xx)
                x_size = xmax - xmin
                if x_size > self.max_size or x_size < self.min_size:
                    continue
                yy = np.random.choice(h_axis, size=2)
                ymin = min(yy)
                ymax = max(yy)
                y_size = ymax - ymin
                if y_size > self.max_size or y_size < self.min_size:
                    continue
                box_in_area = (
                    (boxes[:, 0] >= xmin)
                    & (boxes[:, 1] >= ymin)
                    & (boxes[:, 2] <= xmax)
                    & (boxes[:, 3] <= ymax)
                )
                if len(np.where(box_in_area)[0]) == 0:
                    continue
                im = im[ymin:ymax, xmin:xmax]
                target = target.crop([xmin, ymin, xmax, ymax])
                return Image.fromarray(im), target
            return image, target
        else:
            return image, target


# class RandomCropFixSize(object):
#     def __init__(self, prob, crop_size=512, max_trys=50):
#         self.crop_size = crop_size
#         self.max_trys = max_trys
#         self.prob = prob

#     def __call__(self, image, target):
#         if random.random() < self.prob:
#             im = np.array(image)
#             w, h = image.size
#             h_array = np.zeros((h), dtype=np.int32)
#             w_array = np.zeros((w), dtype=np.int32)
#             boxes = target.bbox.numpy()
#             if len(boxes) == 0:
#                 return image, target
#             for box in boxes:
#                 box = np.round(box, decimals=0).astype(np.int32)
#                 minx = box[0]
#                 maxx = box[2]
#                 w_array[minx:maxx] = 1
#                 miny = box[1]
#                 maxy = box[3]
#                 h_array[miny:maxy] = 1
#             h_axis = np.where(h_array == 0)[0]
#             w_axis = np.where(w_array == 0)[0]
#             if len(h_axis) == 0 or len(w_axis) == 0:
#                 return image, target
#             for _ in range(self.max_trys):
#                 xx = np.random.choice(w_axis, size=2)
#                 xmin = min(xx)
#                 xmax = max(xx)
#                 x_size = xmax - xmin
#                 if x_size > self.max_size or x_size < self.min_size:
#                     continue
#                 yy = np.random.choice(h_axis, size=2)
#                 ymin = min(yy)
#                 ymax = max(yy)
#                 y_size = ymax - ymin
#                 if y_size > self.max_size or y_size < self.min_size:
#                     continue
#                 box_in_area = (
#                     (boxes[:, 0] >= xmin)
#                     & (boxes[:, 1] >= ymin)
#                     & (boxes[:, 2] <= xmax)
#                     & (boxes[:, 3] <= ymax)
#                 )
#                 if len(np.where(box_in_area)[0]) == 0:
#                     continue
#                 im = im[ymin:ymax, xmin:xmax]
#                 target = target.crop([xmin, ymin, xmax, ymax])
#                 return Image.fromarray(im), target
#             return image, target
#         else:
#             return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            image = F.hflip(image)
            target = target.transpose(0)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        return F.to_tensor(image), target


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image, target):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class RandomBrightness(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            brightness_factor = random.uniform(0.5, 2)
            image = F.adjust_brightness(image, brightness_factor)
        return image, target


class RandomContrast(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            contrast_factor = random.uniform(0.5, 2)
            image = F.adjust_contrast(image, contrast_factor)
        return image, target


class RandomHue(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            hue_factor = random.uniform(-0.25, 0.25)
            image = F.adjust_hue(image, hue_factor)
        return image, target


class RandomSaturation(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            saturation_factor = random.uniform(0.5, 2)
            image = F.adjust_saturation(image, saturation_factor)
        return image, target


class RandomGamma(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            gamma_factor = random.uniform(0.5, 2)
            image = F.adjust_gamma(image, gamma_factor)
        return image, target


class RandomRotate(object):
    def __init__(self, prob, max_theta=30, fix_rotate=False):
        self.prob = prob
        self.max_theta = max_theta
        self.fix_rotate = fix_rotate

    def __call__(self, image, target):
        if random.random() < self.prob and target is not None:
            # try:
            if self.fix_rotate:
                delta = 30
            else:
                delta = random.uniform(-1 * self.max_theta, self.max_theta)
            width, height = image.size
            ## get the minimal rect to cover the rotated image
            img_box = [[[0, 0], [width, 0], [width, height], [0, height]]]
            rotated_img_box = _quad2minrect(
                _rotate_polygons(img_box, delta, (width / 2, height / 2))
            )
            r_height = int(
                max(rotated_img_box[0][3], rotated_img_box[0][1])
                - min(rotated_img_box[0][3], rotated_img_box[0][1])
            )
            r_width = int(
                max(rotated_img_box[0][2], rotated_img_box[0][0])
                - min(rotated_img_box[0][2], rotated_img_box[0][0])
            )
            r_height = max(r_height, height + 1)
            r_width = max(r_width, width + 1)

            ## padding im
            im_padding = np.zeros((r_height, r_width, 3))
            start_h, start_w = (
                int((r_height - height) / 2.0),
                int((r_width - width) / 2.0),
            )
            end_h, end_w = start_h + height, start_w + width
            im_padding[start_h:end_h, start_w:end_w, :] = image

            M = cv2.getRotationMatrix2D((r_width / 2, r_height / 2), delta, 1)
            im = cv2.warpAffine(im_padding, M, (r_width, r_height))
            im = Image.fromarray(im.astype(np.uint8))
            target = target.rotate(
                -delta, (r_width / 2, r_height / 2), start_h, start_w
            )
            return im, target
            # except:
            #    return image, target
        else:
            return image, target


def _quad2minrect(boxes):
    ## trans a quad(N*4) to a rectangle(N*4) which has miniual area to cover it
    return np.hstack(
        (
            boxes[:, ::2].min(axis=1).reshape((-1, 1)),
            boxes[:, 1::2].min(axis=1).reshape((-1, 1)),
            boxes[:, ::2].max(axis=1).reshape((-1, 1)),
            boxes[:, 1::2].max(axis=1).reshape((-1, 1)),
        )
    )


def _boxlist2quads(boxlist):
    res = np.zeros((len(boxlist), 8))
    for i, box in enumerate(boxlist):
        # print(box)
        res[i] = np.array(
            [
                box[0][0],
                box[0][1],
                box[1][0],
                box[1][1],
                box[2][0],
                box[2][1],
                box[3][0],
                box[3][1],
            ]
        )
    return res


def _rotate_polygons(polygons, angle, r_c):
    ## polygons: N*8
    ## r_x: rotate center x
    ## r_y: rotate center y
    ## angle: -15~15

    rotate_boxes_list = []
    for poly in polygons:
        box = Polygon(poly)
        rbox = affinity.rotate(box, angle, r_c)
        if len(list(rbox.exterior.coords)) < 5:
            print("img_box_ori:", poly)
            print("img_box_rotated:", rbox)
        # assert(len(list(rbox.exterior.coords))>=5)
        rotate_boxes_list.append(rbox.boundary.coords[:-1])
    res = _boxlist2quads(rotate_boxes_list)
    return res
