# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .coco import COCODataset
from .concat_dataset import ConcatDataset, MixDataset
from .icdar import IcdarDataset
from .scut import ScutDataset
from .synthtext import SynthtextDataset
from .total_text import TotaltextDataset
from .utils import extract_datasets


__all__ = [
    "COCODataset",
    "ConcatDataset",
    "IcdarDataset",
    "SynthtextDataset",
    "MixDataset",
    "ScutDataset",
    "TotaltextDataset",
]
