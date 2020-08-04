#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
# from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.engine.trainer import do_train
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.solver import make_lr_scheduler, make_optimizer
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import get_rank, synchronize, get_world_size

# from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.utils.logging import Logger, setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
# from maskrcnn_benchmark.data.datasets import extract_datasets
from maskrcnn_benchmark.engine.launch import launch
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for multi-precision via apex.amp')


def train(cfg, local_rank, distributed, tb_logger):
    model = build_detection_model(cfg)
    device = torch.device(cfg.MODEL.DEVICE)
    model.to(device)

    optimizer = make_optimizer(cfg, model)
    scheduler = make_lr_scheduler(cfg, optimizer)

    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    model, optimizer = amp.initialize(model, optimizer, opt_level=amp_opt_level)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
        )

    arguments = {}
    arguments["iteration"] = 0

    output_dir = cfg.OUTPUT_DIR

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    extra_checkpoint_data = checkpointer.load(
        cfg.MODEL.WEIGHT, resume=cfg.SOLVER.RESUME
    )
    if cfg.SOLVER.RESUME:
        arguments.update(extra_checkpoint_data)

    data_loader = make_data_loader(
        cfg,
        is_train=True,
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )

    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    do_train(
        model,
        data_loader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        checkpoint_period,
        arguments,
        tb_logger,
        cfg,
        local_rank,
    )

    return model


# def test(cfg, model, distributed):
#     if distributed:
#         model = model.module
#     torch.cuda.empty_cache()  # TODO check if it helps
#     iou_types = ("bbox",)
#     if cfg.MODEL.MASK_ON:
#         iou_types = iou_types + ("segm",)
#     output_folders = [None] * len(cfg.DATASETS.TEST)
#     if cfg.OUTPUT_DIR:
#         dataset_names = cfg.DATASETS.TEST
#         for idx, dataset_name in enumerate(dataset_names):
#             output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
#             mkdir(output_folder)
#             output_folders[idx] = output_folder
#     data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=distributed)
#     for output_folder, data_loader_val in zip(output_folders, data_loaders_val):
#         inference(
#             model,
#             data_loader_val,
#             iou_types=iou_types,
#             box_only=cfg.MODEL.RPN_ONLY,
#             device=cfg.MODEL.DEVICE,
#             expected_results=cfg.TEST.EXPECTED_RESULTS,
#             expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
#             output_folder=output_folder,
#         )
#         synchronize()

# def setup(args):
#     cfg = get_cfg()
#     cfg.merge_from_file(args.config_file)
#     cfg.merge_from_list(args.opts)
#     cfg.freeze()
#     set_global_cfg(cfg.GLOBAL)
#
#     colorful_logging = not args.no_color
#     output_dir = cfg.OUTPUT_DIR
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
#
#     logger = setup_logger(output_dir, color=colorful_logging, distributed_rank=comm.get_rank())
#     logger.info(
#         "Using {} GPUs per machine. Rank of current process: {}".format(
#             args.num_gpus, comm.get_rank()
#         )
#     )
#     logger.info(args)
#
#     logger.info("Environment info:\n" + collect_env_info())
#     logger.info(
#         "Loaded config file {}:\n{}".format(args.config_file, open(args.config_file, "r").read())
#     )
#     logger.info("Running with full config:\n{}".format(cfg))
#     if comm.get_rank() == 0 and output_dir:
#         path = os.path.join(output_dir, "config.yaml")
#         with open(path, "w") as f:
#             f.write(cfg.dump())
#         logger.info("Full config saved to {}".format(os.path.abspath(path)))
#     return cfg


def main(args):
    # parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    # parser.add_argument(
    #     "--config-file",
    #     default="",
    #     metavar="FILE",
    #     help="path to config file",
    #     type=str,
    # )
    # parser.add_argument("--local-rank", type=int, default=0)
    # parser.add_argument(
    #     "--skip-test",
    #     dest="skip_test",
    #     help="Do not test the final model",
    #     action="store_true",
    # )
    # parser.add_argument(
    #     "opts",
    #     help="Modify config options using the command-line",
    #     default=None,
    #     nargs=argparse.REMAINDER,
    # )
    # parser.add_argument(
    #     "--eval-only", action="store_true", help="perform evaluation only"
    # )
    # parser.add_argument(
    #     "--no-color", action="store_true", help="disable colorful logging"
    # )
    # parser.add_argument(
    #     "--num-gpus", type=int, default=1, help="number of gpus per machine"
    # )
    # parser.add_argument("--num-machines", type=int, default=1)
    # parser.add_argument(
    #     "--machine-rank",
    #     type=int,
    #     default=0,
    #     help="the rank of this machine (unique per machine)",
    # )
    # port = 2 ** 15 + 2 ** 14 + hash(os.getuid()) % 2 ** 14
    # parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(port))
    # parser.add_argument(
    #     "opts",
    #     help="Modify config options using the command-line",
    #     default=None,
    #     nargs=argparse.REMAINDER,
    # )
    #
    # args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    # num_gpus = args.num_gpus
    args.distributed = num_gpus > 1
    # args.distributed = get_world_size() > 1
    args.local_rank = get_rank() % args.num_gpus
    
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    # distributed = get_world_size() > 1
    # args.distributed = distributed
    # if distributed:
    #     args.local_rank = get_rank() % args.num_gpus

    print(args.config_file)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(args.num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    tb_logger = Logger(cfg.OUTPUT_DIR, get_rank())
    train(cfg, args.local_rank, args.distributed, tb_logger)

    # if not args.skip_test:
    #     test(cfg, model, args.distributed)


# if __name__ == "__main__":
#     main()

def parse_args(in_args=None):
    """
    Method optionally supports passing arguments. If not provided it is read in
    from sys.argv. If providing it should be a list as per python documentation.
    See https://docs.python.org/3/library/argparse.html#argparse.ArgumentParser.parse_args
    """
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--no-color", action="store_true", help="disable colorful logging")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus per machine")
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid()) % 2 ** 14
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(port))
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(in_args)


def detectron2_launch(args):
    cfg.merge_from_list(args.opts)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    # extract_datasets(cfg)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    detectron2_launch(parse_args())
