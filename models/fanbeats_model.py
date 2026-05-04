# ------------------------------------------------------------------------------------------------------------------------
# FANBEATS: Frequency and Attention-augmented Neural Basis Expansion Analysis for interpretable Time Series forecasting
#
# Copyright (c) 2026 Danish Abbas
#
# This file is part of the FANBEATS project.
# Licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License (CC BY-NC 4.0).
#
# See the LICENSE file in the root directory for full details.
# ------------------------------------------------------------------------------------------------------------------------

"""
models/fanbeats_model.py

"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.fanbeats_modules import (
    UnifiedTemporalAttentionNetwork,
    SpectralFilteringAndAttentionModule,
    MultiscaleWaveletAttentionModule,
    CrossDomainIntegrator,
)

class FANBEATSModel(nn.Module):
    def __init__(self, lookback_window: int, nbeats_model: nn.Module):
        super().__init__()

        self.lookback_window = lookback_window
        self.nbeats = nbeats_model

        self.denoise_threshold = nn.Parameter(torch.tensor(-6.0))
        self.UTAN = UnifiedTemporalAttentionNetwork()
        self.SFAM = SpectralFilteringAndAttentionModule(
            lookback_window=lookback_window,
            attention_module=self.UTAN,
            denoise_threshold=self.denoise_threshold,
        )
        self.MWAM = MultiscaleWaveletAttentionModule(
            lookback_window=lookback_window,
            attention_module=self.UTAN,
            denoise_threshold=self.denoise_threshold,
        )
        self.CDI = CrossDomainIntegrator()

        self.input_gate = nn.Parameter(torch.tensor(0.0))

        self._last_diagnostics = {}

    def get_diagnostics(self):
        return self._last_diagnostics
    
    def get_attention_diagnostics(self):
        return self._last_diagnostics

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor, return_decomposition: bool = False,) -> torch.Tensor:
        original_input = x

        # SFAM
        fft_att, fft_guided, fft_att_weights = self.SFAM(original_input)
        sfam_diag = self.SFAM._last_diag

        # MWAM
        wav_att, wav_att_weights = self.MWAM(original_input)
        mwam_diag = self.MWAM._last_diag

        # CDI
        enhanced = self.CDI(
            fft_att,
            fft_guided,
            wav_att,
            original_input,
        )

        gate = torch.sigmoid(self.input_gate)
        final_input = (1 - gate) * original_input + gate * enhanced

        # passing to modified Backbone
        if return_decomposition:
            backbone_out = self.nbeats(final_input, x_mask, return_decomposition=True)
            forecast = backbone_out["forecast"]
        else:
            backbone_out = None
            forecast = self.nbeats(final_input, x_mask)

        self._last_diagnostics = {
            "forecast": forecast.detach().cpu(),

            "freq_attention_weights": None if fft_att_weights is None else fft_att_weights.detach().cpu(),
            "wav_attention_weights": None if wav_att_weights is None else wav_att_weights.detach().cpu(),

            "sfam_freqs": sfam_diag.get("freqs"),
            "sfam_fft_magnitude": sfam_diag.get("fft_magnitude"),
            "sfam_low_cutoff": sfam_diag.get("low_cutoff"),
            "sfam_high_cutoff": sfam_diag.get("high_cutoff"),
            "sfam_low_mask": sfam_diag.get("low_mask"),
            "sfam_high_mask": sfam_diag.get("high_mask"),
            "sfam_band_mask": sfam_diag.get("band_mask"),
            "sfam_denoise_mask": sfam_diag.get("denoise_mask"),
            "sfam_freq_guided_mask": sfam_diag.get("freq_guided_mask"),
            "sfam_filtered_time_signal": sfam_diag.get("filtered_time_signal"),
            "sfam_fft_attended": sfam_diag.get("x_attended"),
            "sfam_fft_guided": sfam_diag.get("x_guided"),

            "mwam_coefficients": mwam_diag.get("coefficients"),
            "mwam_denoised_coefficients": mwam_diag.get("denoised_coefficients"),
            "mwam_scale_weights": mwam_diag.get("scale_weights"),
            "mwam_raw_wav": mwam_diag.get("raw_wav"),
            "mwam_wav_attended": mwam_diag.get("x_wav_attended"),
            "mwam_mother_wavelet": mwam_diag.get("mother_wavelet"),
            "mwam_mother_wavelet_norm": mwam_diag.get("mother_wavelet_norm"),

            "cdi_weights": self.CDI.get_fusion_weights(),
            "cdi_temperature": self.CDI.get_temperature(),
            "cdi_residual_weight": self.CDI.get_residual_weight(),

            "fusion_gate": gate.detach().cpu(),
            "original_input": original_input.detach().cpu(),
            "enhanced_input": enhanced.detach().cpu(),
            "final_input": final_input.detach().cpu(),
        }

        if return_decomposition and backbone_out is not None:
            self._last_diagnostics.update({
                "trend_forecast": backbone_out["trend_forecast"].detach().cpu(),
                "seasonality_forecast": backbone_out["seasonality_forecast"].detach().cpu(),
                "trend_block_forecasts": backbone_out["trend_block_forecasts"],
                "seasonality_block_forecasts": backbone_out["seasonality_block_forecasts"],
                "generic_block_forecasts": backbone_out["generic_block_forecasts"],
            })

        return forecast