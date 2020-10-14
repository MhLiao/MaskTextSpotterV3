# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import random, cv2, numpy as np
from PIL import Image
from shapely import affinity
from shapely.geometry import Polygon
from torchvision.transforms import functional as F
random.seed(1)
np.random.seed(1)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for t in self.transforms:
            image = t(image)
        return image

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

    def __call__(self, image):
        size = self.get_size(image.size)
        image = F.resize(image, size)

        return image


class ToTensor(object):
    def __call__(self, image):
        return F.to_tensor(image)


class Normalize(object):
    def __init__(self, mean, std, to_bgr255=True):
        self.mean = mean
        self.std = std
        self.to_bgr255 = to_bgr255

    def __call__(self, image):
        if self.to_bgr255:
            image = image[[2, 1, 0]] * 255
        return image


class RandomBrightness(object):
    def __init__(self, prob=1.):
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            brightness_factor = random.uniform(0.5, 2)
            image = F.adjust_brightness(image, brightness_factor)
        return image

class RandomContrast(object):
    def __init__(self, prob=1.):
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            contrast_factor = random.uniform(0.5, 2)
            image = F.adjust_contrast(image, contrast_factor)
        return image


class RandomHue(object):
    def __init__(self, prob=1.):
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            hue_factor = random.uniform(-0.25, 0.25)
            image = F.adjust_hue(image, hue_factor)
        return image


class RandomSaturation(object):
    def __init__(self, prob=1.):
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            saturation_factor = random.uniform(0.5, 2)
            image = F.adjust_saturation(image, saturation_factor)
        return image


class RandomGamma(object):
    def __init__(self, prob=1.):
        self.prob = prob

    def __call__(self, image):
        if random.random() < self.prob:
            gamma_factor = random.uniform(0.5, 2)
            image = F.adjust_gamma(image, gamma_factor)
        return image


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
