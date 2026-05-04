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

"""
modified N-BEATS Model.
"""
from typing import Tuple

import numpy as np
import torch as t

class NBeatsBlock(t.nn.Module):
    def __init__(self,
                 input_size,
                 theta_size: int,
                 basis_function: t.nn.Module,
                 layers: int,
                 layer_size: int):
        super().__init__()
        self.input_size = input_size

        self.temporal_block = t.nn.Sequential(
            t.nn.Conv1d(1, 8, kernel_size=3, padding=1),
            t.nn.ReLU(),
            t.nn.Conv1d(8, 1, kernel_size=3, padding=1),
        )

        self.layers = t.nn.ModuleList(
            [t.nn.Linear(in_features=input_size, out_features=layer_size)] +
            [t.nn.Linear(in_features=layer_size, out_features=layer_size)
             for _ in range(layers - 1)]
        )

        self.basis_parameters = t.nn.Linear(in_features=layer_size, out_features=theta_size)
        self.basis_function = basis_function

        self.norm = t.nn.LayerNorm(layer_size)

        self.activation = t.nn.SiLU()

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        x_conv = self.temporal_block(x.unsqueeze(1)).squeeze(1)
        block_input = x + x_conv                                 

        for layer in self.layers:
            block_input = self.activation(layer(block_input))
            block_input = self.norm(block_input)

        basis_parameters = self.basis_parameters(block_input)
        return self.basis_function(basis_parameters)


class NBeats(t.nn.Module):

    def __init__(self, blocks: t.nn.ModuleList, block_types=None):
        super().__init__()
        self.blocks = blocks
        self.block_types = block_types if block_types is not None else ["generic"] * len(blocks)

    def forward(
        self,
        x: t.Tensor,
        input_mask: t.Tensor,
        return_decomposition: bool = False,
    ):
        residuals = x.flip(dims=(1,))
        input_mask = input_mask.flip(dims=(1,))
        forecast = x[:, -1:]

        if return_decomposition:
            batch_size = x.size(0)
            forecast_size = forecast.size(1)

            trend_forecast = t.zeros(
                (batch_size, forecast_size),
                device=x.device,
                dtype=x.dtype,
            )
            seasonality_forecast = t.zeros(
                (batch_size, forecast_size),
                device=x.device,
                dtype=x.dtype,
            )

            trend_block_forecasts = []
            seasonality_block_forecasts = []
            generic_block_forecasts = []

        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals)
            residuals = (residuals - backcast) * input_mask
            forecast = forecast + block_forecast

            if return_decomposition:
                block_type = self.block_types[i] if i < len(self.block_types) else "generic"

                if block_type == "trend":
                    trend_forecast = trend_forecast + block_forecast
                    trend_block_forecasts.append(block_forecast.detach().cpu())
                elif block_type == "seasonality":
                    seasonality_forecast = seasonality_forecast + block_forecast
                    seasonality_block_forecasts.append(block_forecast.detach().cpu())
                else:
                    generic_block_forecasts.append(block_forecast.detach().cpu())

        if not return_decomposition:
            return forecast

        decomposition = {
            "forecast": forecast,
            "trend_forecast": trend_forecast,
            "seasonality_forecast": seasonality_forecast,
            "trend_block_forecasts": trend_block_forecasts,
            "seasonality_block_forecasts": seasonality_block_forecasts,
            "generic_block_forecasts": generic_block_forecasts,
        }
        return decomposition


class GenericBasis(t.nn.Module):

    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size

    def forward(self, theta: t.Tensor):
        return theta[:, :self.backcast_size], theta[:, -self.forecast_size:]


class TrendBasis(t.nn.Module):
    def __init__(self, degree_of_polynomial: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.polynomial_size = degree_of_polynomial + 1
        self.backcast_time = t.nn.Parameter(
            t.tensor(np.concatenate([
                np.power(np.arange(backcast_size, dtype=float) / backcast_size, i)[None, :]
                for i in range(self.polynomial_size)
            ]), dtype=t.float32),
            requires_grad=False
        )
        self.forecast_time = t.nn.Parameter(
            t.tensor(np.concatenate([
                np.power(np.arange(forecast_size, dtype=float) / forecast_size, i)[None, :]
                for i in range(self.polynomial_size)
            ]), dtype=t.float32),
            requires_grad=False
        )

        self.trend_scale = t.nn.Parameter(t.ones(1))

    def forward(self, theta: t.Tensor):
        backcast = t.einsum('bp,pt->bt', theta[:, self.polynomial_size:], self.backcast_time)
        forecast = t.einsum('bp,pt->bt', theta[:, :self.polynomial_size], self.forecast_time)

        eps = 1e-6
        backcast_std = backcast.std(dim=1, keepdim=True, unbiased=False)
        forecast_std = forecast.std(dim=1, keepdim=True, unbiased=False)

        backcast = backcast / (backcast_std + eps)
        forecast = forecast / (forecast_std + eps)

        backcast = self.trend_scale * backcast
        forecast = self.trend_scale * forecast

        return backcast, forecast

class SeasonalityBasis(t.nn.Module):

    def __init__(self, harmonics: int, backcast_size: int, forecast_size: int):
        super().__init__()
        self.frequency = np.append(
            np.zeros(1, dtype=np.float32),
            np.arange(harmonics, harmonics / 2 * forecast_size, dtype=np.float32) / harmonics
        )[None, :]

        backcast_grid = -2 * np.pi * (
            np.arange(backcast_size, dtype=np.float32)[:, None] / forecast_size
        ) * self.frequency
        forecast_grid = 2 * np.pi * (
            np.arange(forecast_size, dtype=np.float32)[:, None] / forecast_size
        ) * self.frequency

        self.backcast_cos_template = t.nn.Parameter(
            t.tensor(np.transpose(np.cos(backcast_grid)), dtype=t.float32),
            requires_grad=False
        )
        self.backcast_sin_template = t.nn.Parameter(
            t.tensor(np.transpose(np.sin(backcast_grid)), dtype=t.float32),
            requires_grad=False
        )
        self.forecast_cos_template = t.nn.Parameter(
            t.tensor(np.transpose(np.cos(forecast_grid)), dtype=t.float32),
            requires_grad=False
        )
        self.forecast_sin_template = t.nn.Parameter(
            t.tensor(np.transpose(np.sin(forecast_grid)), dtype=t.float32),
            requires_grad=False
        )

    def forward(self, theta: t.Tensor):
        params_per_harmonic = theta.shape[1] // 4

        backcast_harmonics_cos = t.einsum(
            'bp,pt->bt',
            theta[:, 2 * params_per_harmonic:3 * params_per_harmonic],
            self.backcast_cos_template
        )
        backcast_harmonics_sin = t.einsum(
            'bp,pt->bt',
            theta[:, 3 * params_per_harmonic:],
            self.backcast_sin_template
        )
        backcast = backcast_harmonics_sin + backcast_harmonics_cos

        forecast_harmonics_cos = t.einsum(
            'bp,pt->bt',
            theta[:, :params_per_harmonic],
            self.forecast_cos_template
        )
        forecast_harmonics_sin = t.einsum(
            'bp,pt->bt',
            theta[:, params_per_harmonic:2 * params_per_harmonic],
            self.forecast_sin_template
        )
        forecast = forecast_harmonics_sin + forecast_harmonics_cos

        eps = 1e-6
        backcast = backcast / (backcast.std(dim=1, keepdim=True, unbiased=False) + eps)
        forecast = forecast / (forecast.std(dim=1, keepdim=True, unbiased=False) + eps)

        return backcast, forecast