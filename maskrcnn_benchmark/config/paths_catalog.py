# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""Centralized catalog of paths."""

import os


class DatasetCatalog(object):
    DATA_DIR = "datasets"
    # DATA_DIR = "/share/mhliao/MaskTextSpotterV3/datasets/"

    DATASETS = {
        "coco_2014_train": (
            "coco/train2014",
            "coco/annotations/instances_train2014.json",
        ),
        "coco_2014_val": ("coco/val2014", "coco/annotations/instances_val2014.json"),
        "coco_2014_minival": (
            "coco/val2014",
            "coco/annotations/instances_minival2014.json",
        ),
        "coco_2014_valminusminival": (
            "coco/val2014",
            "coco/annotations/instances_valminusminival2014.json",
        ),
        "icdar_2013_train": ("icdar2013/train_images", "icdar2013/train_gts"),
        "icdar_2013_test": ("icdar2013/test_images", "icdar2013/test_gts"),
        "rotated_ic13_test_0": ("icdar2013/rotated_test_images_0", "icdar2013/rotated_test_gts_0"),
        "rotated_ic13_test_15": ("icdar2013/rotated_test_images_15", "icdar2013/rotated_test_gts_15"),
        "rotated_ic13_test_30": ("icdar2013/rotated_test_images_30", "icdar2013/rotated_test_gts_30"),
        "rotated_ic13_test_45": ("icdar2013/rotated_test_images_45", "icdar2013/rotated_test_gts_45"),
        "rotated_ic13_test_60": ("icdar2013/rotated_test_images_60", "icdar2013/rotated_test_gts_60"),
        "rotated_ic13_test_75": ("icdar2013/rotated_test_images_75", "icdar2013/rotated_test_gts_75"),
        "rotated_ic13_test_85": ("icdar2013/rotated_test_images_85", "icdar2013/rotated_test_gts_85"),
        "rotated_ic13_test_90": ("icdar2013/rotated_test_images_90", "icdar2013/rotated_test_gts_90"),
        "rotated_ic13_test_-15": ("icdar2013/rotated_test_images_-15", "icdar2013/rotated_test_gts_-15"),
        "rotated_ic13_test_-30": ("icdar2013/rotated_test_images_-30", "icdar2013/rotated_test_gts_-30"),
        "rotated_ic13_test_-45": ("icdar2013/rotated_test_images_-45", "icdar2013/rotated_test_gts_-45"),
        "rotated_ic13_test_-60": ("icdar2013/rotated_test_images_-60", "icdar2013/rotated_test_gts_-60"),
        "rotated_ic13_test_-75": ("icdar2013/rotated_test_images_-75", "icdar2013/rotated_test_gts_-75"),
        "rotated_ic13_test_-90": ("icdar2013/rotated_test_images_-90", "icdar2013/rotated_test_gts_-90"),
        "icdar_2015_train": ("icdar2015/train_images", "icdar2015/train_gts"),
        "icdar_2015_test": (
            "icdar2015/test_images",
            # "icdar2015/test_gts",
        ),
        "synthtext_train": ("synthtext/train_images", "synthtext/train_gts"),
        "synthtext_test": ("synthtext/test_images", "synthtext/test_gts"),
        "total_text_train": ("total_text/train_images", "total_text/train_gts"),
        "td500_train": ("TD_TR/TD500/train_images", "TD500/train_gts"),
        "td500_test": ("TD_TR/TD500/test_images", ),
        "tr400_train": ("TD_TR/TR400/train_images", "TR400/train_gts"),
        "total_text_test": (
            "total_text/test_images",
            # "total_text/test_gts",
        ),
        "scut-eng-char_train": (
            "scut-eng-char/train_images",
            "scut-eng-char/train_gts",
        ),
    }

    @staticmethod
    def get(name):
        if "coco" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                root=os.path.join(data_dir, attrs[0]),
                ann_file=os.path.join(data_dir, attrs[1]),
            )
            return dict(factory="COCODataset", args=args)
        elif "icdar_2013" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                use_charann=True,
                imgs_dir=os.path.join(data_dir, attrs[0]),
                gts_dir=os.path.join(data_dir, attrs[1]),
                # imgs_dir='/tmp/icdar2013/icdar2013/train_images',
                # gts_dir='/tmp/icdar2013/icdar2013/train_gts',
            )
            return dict(args=args, factory="IcdarDataset")
        elif "rotated_ic13" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                use_charann=True,
                imgs_dir=os.path.join(data_dir, attrs[0]),
                gts_dir=os.path.join(data_dir, attrs[1]),
            )
            return dict(args=args, factory="IcdarDataset")
        elif "icdar_2015" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            if len(attrs) > 1:
                gts_dir = os.path.join(data_dir, attrs[1])
            else:
                gts_dir = None

            args = dict(
                use_charann=False,
                imgs_dir=os.path.join(data_dir, attrs[0]),
                gts_dir=gts_dir,
                # imgs_dir='/tmp/icdar2015/icdar2015/train_images/',
                # gts_dir='/tmp/icdar2015/icdar2015/train_gts/',
            )
            return dict(args=args, factory="IcdarDataset")
        elif "synthtext" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                use_charann=True,
                list_file_path=os.path.join(data_dir, "synthtext/train_list.txt"),
                imgs_dir=os.path.join(data_dir, attrs[0]),
                gts_dir=os.path.join(data_dir, attrs[1]),
                # imgs_dir='/tmp/synth/SynthText/',
                # gts_dir='/tmp/synth_gt/SynthText_GT_E2E/',
            )
            return dict(args=args, factory="SynthtextDataset")
        elif "total_text" in name:
            data_dir = DatasetCatalog.DATA_DIR
            # data_dir = '/tmp/total_text/'
            attrs = DatasetCatalog.DATASETS[name]
            if len(attrs) > 1:
                gts_dir = os.path.join(data_dir, attrs[1])
            else:
                gts_dir = None
            args = dict(
                use_charann=False,
                imgs_dir=os.path.join(data_dir, attrs[0]),
                gts_dir=gts_dir,
                # imgs_dir='/tmp/total_text/total_text/train_images/',
                # gts_dir='/tmp/total_text/total_text/train_gts/',
            )
            return dict(args=args, factory="TotaltextDataset")
        elif "scut-eng-char" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            args = dict(
                use_charann=True,
                imgs_dir=os.path.join(data_dir, attrs[0]),
                gts_dir=os.path.join(data_dir, attrs[1]),
                # imgs_dir='/tmp/scut-eng-char/scut-eng-char/train_images/',
                # gts_dir='/tmp/scut-eng-char/scut-eng-char/train_gts/',
            )
            return dict(args=args, factory="ScutDataset")
        elif "td500" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            if len(attrs) > 1:
                gts_dir = os.path.join(data_dir, attrs[1])
            else:
                gts_dir = None
            args = dict(
                use_charann=False,
                imgs_dir=os.path.join(data_dir, attrs[0]),
                gts_dir=gts_dir,
            )
            return dict(args=args, factory="TotaltextDataset")
        elif "tr400" in name:
            data_dir = DatasetCatalog.DATA_DIR
            attrs = DatasetCatalog.DATASETS[name]
            if len(attrs) > 1:
                gts_dir = os.path.join(data_dir, attrs[1])
            else:
                gts_dir = None
            args = dict(
                use_charann=False,
                imgs_dir=os.path.join(data_dir, attrs[0]),
                gts_dir=gts_dir,
            )
            return dict(args=args, factory="TotaltextDataset")
        raise RuntimeError("Dataset not available: {}".format(name))


class ModelCatalog(object):
    S3_C2_DETECTRON_URL = "https://dl.fbaipublicfiles.com/detectron"
    C2_IMAGENET_MODELS = {
        'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
        'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
        "MSRA/R-50": "ImageNetPretrained/MSRA/R-50.pkl",
        "MSRA/R-50-GN": "ImageNetPretrained/47261647/R-50-GN.pkl",
        "MSRA/R-101": "ImageNetPretrained/MSRA/R-101.pkl",
        "MSRA/R-101-GN": "ImageNetPretrained/47592356/R-101-GN.pkl",
        "FAIR/20171220/X-101-32x8d": "ImageNetPretrained/20171220/X-101-32x8d.pkl",
    }

    C2_DETECTRON_SUFFIX = "output/train/{}coco_2014_train%3A{}coco_2014_valminusminival/generalized_rcnn/model_final.pkl"
    C2_DETECTRON_MODELS = {
        "35857197/e2e_faster_rcnn_R-50-C4_1x": "01_33_49.iAX0mXvW",
        "35857345/e2e_faster_rcnn_R-50-FPN_1x": "01_36_30.cUF7QR7I",
        "35857890/e2e_faster_rcnn_R-101-FPN_1x": "01_38_50.sNxI7sX7",
        "36761737/e2e_faster_rcnn_X-101-32x8d-FPN_1x": "06_31_39.5MIHi1fZ",
        "35858791/e2e_mask_rcnn_R-50-C4_1x": "01_45_57.ZgkA7hPB",
        "35858933/e2e_mask_rcnn_R-50-FPN_1x": "01_48_14.DzEQe4wC",
        "35861795/e2e_mask_rcnn_R-101-FPN_1x": "02_31_37.KqyEK4tT",
        "36761843/e2e_mask_rcnn_X-101-32x8d-FPN_1x": "06_35_59.RZotkLKI",
        "37129812/e2e_mask_rcnn_X-152-32x8d-FPN-IN5k_1.44x": "09_35_36.8pzTQKYK",
        # keypoints
        "37697547/e2e_keypoint_rcnn_R-50-FPN_1x": "08_42_54.kdzV35ao"
    }

    @staticmethod
    def get(name):
        if name.startswith("Caffe2Detectron/COCO"):
            return ModelCatalog.get_c2_detectron_12_2017_baselines(name)
        if name.startswith("ImageNetPretrained"):
            return ModelCatalog.get_c2_imagenet_pretrained(name)
        raise RuntimeError("model not present in the catalog {}".format(name))

    @staticmethod
    def get_c2_imagenet_pretrained(name):
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        name = name[len("ImageNetPretrained/") :]
        name = ModelCatalog.C2_IMAGENET_MODELS[name]
        if 'resnet34' in name or 'resnet18' in name:
            return name
        url = "/".join([prefix, name])
        return url

    @staticmethod
    def get_c2_detectron_12_2017_baselines(name):
        # Detectron C2 models are stored following the structure
        # prefix/<model_id>/2012_2017_baselines/<model_name>.yaml.<signature>/suffix
        # we use as identifiers in the catalog Caffe2Detectron/COCO/<model_id>/<model_name>
        prefix = ModelCatalog.S3_C2_DETECTRON_URL
        suffix = ModelCatalog.C2_DETECTRON_SUFFIX
        # remove identification prefix
        name = name[len("Caffe2Detectron/COCO/") :]
        # split in <model_id> and <model_name>
        model_id, model_name = name.split("/")
        # parsing to make it match the url address from the Caffe2 models
        model_name = "{}.yaml".format(model_name)
        signature = ModelCatalog.C2_DETECTRON_MODELS[name]
        unique_name = ".".join([model_name, signature])
        url = "/".join([prefix, model_id, "12_2017_baselines", unique_name, suffix])
        return url
