#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Train a classification model."""

import pycls.core.config as config
import pycls.core.distributed as dist
import pycls.core.trainer as trainer
from pycls.core.config import cfg

from template_lib.v2.config import global_cfg
from template_lib.d2.utils import D2Utils

def main():
    config.load_cfg_fom_args("Train a classification model.")
    config.assert_and_infer_cfg()

    D2Utils.cfg_merge_from_easydict(cfg, global_cfg)

    cfg.freeze()
    dist.multi_proc_run(num_proc=cfg.NUM_GPUS, fun=trainer.train_model)


if __name__ == "__main__":
    main()
