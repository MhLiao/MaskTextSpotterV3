# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
import sys

from tensorboardX import SummaryWriter


def setup_logger(name, save_dir, distributed_rank=0):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, "log.txt"))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


class Logger(object):
    def __init__(self, log_dir, distributed_rank=0):
        """Create a summary writer logging to log_dir."""
        self.distributed_rank = distributed_rank
        if distributed_rank == 0:
            self.writer = SummaryWriter(log_dir)


    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        if self.distributed_rank == 0:
            self.writer.add_scalar(tag, value, step)
