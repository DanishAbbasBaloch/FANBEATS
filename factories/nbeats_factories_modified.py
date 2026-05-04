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
# - updates to the N-BEATS architecture
# - integration with FANBEATS components
# - adaptation of training pipeline
#
# Copyright (c) 2026 Danish Abbas
# Licensed under CC BY-NC 4.0 (see root LICENSE file).

import gin
import numpy as np
import torch as t

from models.nbeats_modified import GenericBasis, NBeats, NBeatsBlock, SeasonalityBasis, TrendBasis


@gin.configurable()
def interpretable(input_size: int,
                  output_size: int,
                  trend_blocks: int,
                  trend_layers: int,
                  trend_layer_size: int,
                  degree_of_polynomial: int,
                  seasonality_blocks: int,
                  seasonality_layers: int,
                  seasonality_layer_size: int,
                  num_of_harmonics: int):
    """
    Create N-BEATS interpretable model.
    """
    trend_block = NBeatsBlock(
        input_size=input_size,
        theta_size=2 * (degree_of_polynomial + 1),
        basis_function=TrendBasis(
            degree_of_polynomial=degree_of_polynomial,
            backcast_size=input_size,
            forecast_size=output_size
        ),
        layers=trend_layers,
        layer_size=trend_layer_size
    )

    seasonality_block = NBeatsBlock(
        input_size=input_size,
        theta_size=4 * int(
            np.ceil(num_of_harmonics / 2 * output_size) - (num_of_harmonics - 1)
        ),
        basis_function=SeasonalityBasis(
            harmonics=num_of_harmonics,
            backcast_size=input_size,
            forecast_size=output_size
        ),
        layers=seasonality_layers,
        layer_size=seasonality_layer_size
    )

    blocks = (
        [trend_block for _ in range(trend_blocks)]
        + [seasonality_block for _ in range(seasonality_blocks)]
    )

    block_types = ["trend"] * trend_blocks + ["seasonality"] * seasonality_blocks

    return NBeats(
        t.nn.ModuleList(blocks),
        block_types=block_types,
    )
    
@gin.configurable()
def generic(input_size: int, output_size: int,
            stacks: int, layers: int, layer_size: int):
    blocks = [
        NBeatsBlock(
            input_size=input_size,
            theta_size=input_size + output_size,
            basis_function=GenericBasis(
                backcast_size=input_size,
                forecast_size=output_size
            ),
            layers=layers,
            layer_size=layer_size
        )
        for _ in range(stacks)
    ]

    return NBeats(
        t.nn.ModuleList(blocks),
        block_types=["generic"] * stacks,
    )
