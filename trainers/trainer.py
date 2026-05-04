# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

# ------------------------------------------------------------
# Modification Notice (FANBEATS)
# ------------------------------------------------------------
# This file has been modified from the original N-BEATS implementation
# (Oreshkin et al., 2020) for use in the FANBEATS framework.
#
# Modifications include:
# - integration with FANBEATS training pipeline
# - incorporation of hybrid loss formulation
#
# Copyright (c) 2026 Danish Abbas
# Licensed under CC BY-NC 4.0 (see root LICENSE file).

"""
models training logic.
"""
from typing import Iterator

import gin
import numpy as np
import torch as t
from torch import optim

from utils.ops import default_device, to_tensor


@gin.configurable
def trainer(
            model: t.nn.Module,
            training_set: Iterator,
            timeseries_frequency: int,
            loss_name: str,
            iterations: int,
            learning_rate: float = 0.001):

    model = model.to(default_device())
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4   
    )

    scheduler = t.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=iterations
    )

    training_loss_fn = __loss_fn(loss_name)


    for i in range(1, iterations + 1):
        model.train()
        x, x_mask, y, y_mask = map(to_tensor, next(training_set))
        optimizer.zero_grad()
        forecast = model(x, x_mask)
        training_loss = training_loss_fn(x, timeseries_frequency, forecast, y, y_mask)

        if np.isnan(float(training_loss)):
            break

        training_loss.backward()
        t.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()  

    return model

# 1. for hybrid loss
def __loss_fn(loss_name: str):
    def loss(x, freq, forecast, target, target_mask):

        if loss_name == 'HYBRID':
            B, H = forecast.shape
            weights = t.linspace(1.0, 2.0, H).to(forecast.device)
            weights = weights / weights.mean()
            mse = t.mean(weights * (forecast - target) ** 2)

            vx = forecast - t.mean(forecast, dim=1, keepdim=True)
            vy = target - t.mean(target, dim=1, keepdim=True)
            corr = t.sum(vx * vy, dim=1) / (
                t.sqrt(t.sum(vx ** 2, dim=1) + 1e-8) *
                t.sqrt(t.sum(vy ** 2, dim=1) + 1e-8)
            )
            corr_loss = 1 - t.mean(corr)

            return mse + 0.3 * corr_loss

        else:
            raise Exception(f'Unknown loss function: {loss_name}')

    return loss