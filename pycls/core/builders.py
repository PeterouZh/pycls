#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Model and loss construction functions."""

import torch
from pycls.core.config import cfg
from pycls.models.anynet import AnyNet
from pycls.models.effnet import EffNet
from pycls.models.regnet import RegNet
from pycls.models.resnet import ResNet

from exp.cmconv_cls.model_resnet_cmconv_cls import ResNetCMConvCls
from exp.cmconv_cls.loss_cmconv_ce import CMConvCE
from exp.cmconv_cls.model_resnet_smconv_cls import ResNetSMConv
from exp.cmconv_cls.model_resnet_smconv_with_loss import ResNetSMConvLoss
from exp.random_multiple_label.loss_random_multiple_ce import Random_MultiLabel_CE
from exp.transformer_cnn.transformer_resnet import TResNet

# Supported models
_models = {"anynet": AnyNet, "effnet": EffNet, "resnet": ResNet, "regnet": RegNet,
           "resnet_cmconv_cls": ResNetCMConvCls,
           'resnet_smconv': ResNetSMConv,
           'resnet_smconv_with_loss': ResNetSMConvLoss,
           'transformer_resnet': TResNet,
           }

# Supported loss functions
_loss_funs = {"cross_entropy": torch.nn.CrossEntropyLoss,
              "cmconv_cross_entropy": CMConvCE,
              "random_multilabel_ce": Random_MultiLabel_CE}


def get_model():
    """Gets the model class specified in the config."""
    err_str = "Model type '{}' not supported"
    assert cfg.MODEL.TYPE in _models.keys(), err_str.format(cfg.MODEL.TYPE)
    return _models[cfg.MODEL.TYPE]


def get_loss_fun():
    """Gets the loss function class specified in the config."""
    err_str = "Loss function type '{}' not supported"
    assert cfg.MODEL.LOSS_FUN in _loss_funs.keys(), err_str.format(cfg.TRAIN.LOSS)
    return _loss_funs[cfg.MODEL.LOSS_FUN]


def build_model():
    """Builds the model."""
    return get_model()()


def build_loss_fun():
    """Build the loss function."""
    return get_loss_fun()()


def register_model(name, ctor):
    """Registers a model dynamically."""
    _models[name] = ctor


def register_loss_fun(name, ctor):
    """Registers a loss function dynamically."""
    _loss_funs[name] = ctor
