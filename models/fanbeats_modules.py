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
models/fanbeats_modules.py

Core pre-backbone FANBEATS modules for solar wind speed forecasting.

Modules:
- UnifiedTemporalAttentionNetwork (UTAN)
- LowRankRefine
- SpectralFilteringAndAttentionModule (SFAM)
- MultiscaleWaveletAttentionModule (MWAM)
- CrossDomainIntegrator (CDI)

These modules are designed to process a univariate input window

and return enhanced representations , which can then be
fed into the modified interpretable N-BEATS backbone.

"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class UnifiedTemporalAttentionNetwork(nn.Module):

    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.directions = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.attn_fc = nn.Linear(hidden_dim * self.directions, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
 
        if x.ndim != 2:
            raise ValueError(f"UTAN expected input shape [B, L], got {tuple(x.shape)}")

        x_in = x.unsqueeze(-1)  # [B, L, 1]
        rnn_out, _ = self.gru(x_in)  # [B, L, H*dir]
        scores = self.attn_fc(rnn_out).squeeze(-1)  # [B, L]
        attn_weights = torch.softmax(scores, dim=1)
        return attn_weights


class LowRankRefine(nn.Module):

    def __init__(self, lookback_window: int, rank: int = 16) -> None:
        super().__init__()

        if rank <= 0:
            raise ValueError("rank must be positive")

        self.fc1 = nn.Linear(lookback_window, rank)
        self.fc2 = nn.Linear(rank, lookback_window)

    def forward(self, att_w: torch.Tensor) -> torch.Tensor:
        if att_w.ndim != 2:
            raise ValueError(f"LowRankRefine expected [B, L], got {tuple(att_w.shape)}")

        h = F.relu(self.fc1(att_w))
        delta = self.fc2(h)
        return torch.sigmoid(att_w + delta)


class SpectralFilteringAndAttentionModule(nn.Module):

    def __init__(
        self,
        lookback_window: int,
        attention_module: UnifiedTemporalAttentionNetwork,
        denoise_threshold: nn.Parameter,
        beta: float = 10.0,
        selector_dropout: float = 0.3,
        rank: int = 16,
    ) -> None:
        super().__init__()

        self.lookback_window = lookback_window
        self.attention_module = attention_module
        self.denoise_threshold = denoise_threshold
        self.beta = beta

        self.freq_selector = nn.Sequential(
            nn.Linear(lookback_window, 256),
            nn.ReLU(),
            nn.Dropout(selector_dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(selector_dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(selector_dropout),
            nn.Linear(64, 2),
            nn.Sigmoid(),
        )

        self.freq_attention = LowRankRefine(
            lookback_window=lookback_window,
            rank=rank,
        )

        self._last_diag: Dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if x.ndim != 2:
            raise ValueError(f"SFAM expected input shape [B, L], got {tuple(x.shape)}")

        batch_size, lookback_window = x.shape
        device = x.device

        fft_transformed = torch.fft.fft(x, dim=1)
        magnitudes = torch.abs(fft_transformed)
        freqs = torch.fft.fftfreq(lookback_window, d=1.0).to(device)  

        cutoffs = self.freq_selector(x) 
        low_c = cutoffs[:, 0].unsqueeze(1)  
        high_c = cutoffs[:, 1].unsqueeze(1)  

        low_c, high_c = torch.minimum(low_c, high_c), torch.maximum(low_c, high_c)

        bf = freqs.unsqueeze(0).expand(batch_size, -1) 
        abs_bf = torch.abs(bf)

        low_mask = torch.sigmoid((abs_bf - low_c) * self.beta)
        high_mask = 1.0 - torch.sigmoid((abs_bf - high_c) * self.beta)
        band_mask = low_mask * high_mask

        denoise_thr = torch.mean(magnitudes, dim=1, keepdim=True) * torch.exp(self.denoise_threshold)
        denoise_mask = torch.sigmoid((magnitudes - denoise_thr.expand_as(magnitudes)) * self.beta)

        filtered_fft = fft_transformed * band_mask * denoise_mask
        raw_freq = torch.fft.ifft(filtered_fft, dim=1).real  

        att_weights = self.attention_module(raw_freq)  
        x_attended = raw_freq * att_weights

        freq_w = self.freq_attention(att_weights) 
        x_guided = torch.fft.ifft(
            torch.fft.fft(raw_freq, dim=1) * freq_w,
            dim=1,
        ).real

        self._last_diag = {
            "freqs": freqs.detach().cpu(),
            "fft_magnitude": magnitudes.detach().cpu(),
            "low_cutoff": low_c.detach().cpu(),
            "high_cutoff": high_c.detach().cpu(),
            "low_mask": low_mask.detach().cpu(),
            "high_mask": high_mask.detach().cpu(),
            "band_mask": band_mask.detach().cpu(),
            "denoise_mask": denoise_mask.detach().cpu(),
            "freq_guided_mask": freq_w.detach().cpu(),
            "filtered_time_signal": raw_freq.detach().cpu(),
            "x_attended": x_attended.detach().cpu(),
            "x_guided": x_guided.detach().cpu(),
        }

        return x_attended, x_guided, att_weights


class MultiscaleWaveletAttentionModule(nn.Module):

    def __init__(
        self,
        lookback_window: int,
        attention_module: UnifiedTemporalAttentionNetwork,
        denoise_threshold: nn.Parameter,
        scales: int = 4,
        mother_size: int = 10,
    ) -> None:
        super().__init__()

        self.lookback_window = lookback_window
        self.attention_module = attention_module
        self.denoise_threshold = denoise_threshold
        self.scales = scales
        self.mother_size = mother_size

        self.mother_wavelet = nn.Parameter(torch.randn(mother_size) * 0.1)
        self.wavelet_weights = nn.Parameter(torch.ones(scales) / scales)

        self._last_diag: Dict[str, torch.Tensor] = {}

    def _normalized_mother_wavelet(self) -> torch.Tensor:
        denom = torch.sqrt(torch.sum(self.mother_wavelet ** 2) + 1e-12)
        return self.mother_wavelet / denom

    def apply_wavelet_transform(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, length = x.shape
        device = x.device

        wavelet = self._normalized_mother_wavelet()
        coefficients = torch.zeros((batch_size, self.scales, length), device=device)

        fft_sig = torch.fft.fft(x, dim=1)

        for i in range(self.scales):
            scale_factor = 2 ** i
            if scale_factor > 1:
                size = min(self.mother_size * scale_factor, length)
                scaled = F.interpolate(
                    wavelet.view(1, 1, -1),
                    size=int(size),
                    mode="linear",
                    align_corners=False,
                ).view(-1)
            else:
                scaled = wavelet

            padded = torch.zeros(length, device=device)
            padded[: scaled.size(0)] = scaled
            fft_wave = torch.fft.fft(padded)

            coefficients[:, i, :] = torch.fft.ifft(fft_sig * fft_wave, dim=1).real

        return coefficients

    def wavelet_denoise(self, coefficients: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:

        mean_abs = torch.mean(torch.abs(coefficients), dim=2, keepdim=True)
        adaptive_threshold = mean_abs * threshold
        return torch.sign(coefficients) * F.relu(torch.abs(coefficients) - adaptive_threshold)

    def inverse_wavelet_transform(self, coefficients: torch.Tensor) -> torch.Tensor:
 
        batch_size, scales, length = coefficients.shape
        device = coefficients.device

        weights = torch.softmax(self.wavelet_weights, dim=0)
        reconstructed = torch.zeros((batch_size, length), device=device)

        for i in range(scales):
            scale_factor = 2 ** i

            if scale_factor > 1:
                size = min(self.mother_size * scale_factor, length)
                wave = F.interpolate(
                    self.mother_wavelet.view(1, 1, -1),
                    size=int(size),
                    mode="linear",
                    align_corners=False,
                ).view(-1)
            else:
                wave = self.mother_wavelet

            padded = torch.zeros(length, device=device)
            padded[: wave.size(0)] = wave

            fft_wave = torch.fft.fft(padded)
            fft_coeffs = torch.fft.fft(coefficients[:, i, :], dim=1)

            reconstructed += weights[i] * torch.fft.ifft(
                fft_coeffs / (fft_wave + 1e-10),
                dim=1,
            ).real

        return reconstructed

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 2:
            raise ValueError(f"MWAM expected input shape [B, L], got {tuple(x.shape)}")

        coefficients = self.apply_wavelet_transform(x)
        threshold = torch.exp(self.denoise_threshold)
        denoised = self.wavelet_denoise(coefficients, threshold)

        raw_wav = self.inverse_wavelet_transform(denoised)
        wav_att_weights = self.attention_module(raw_wav)
        x_wav_attended = raw_wav * wav_att_weights

        self._last_diag = {
            "coefficients": coefficients.detach().cpu(),
            "denoised_coefficients": denoised.detach().cpu(),
            "scale_weights": torch.softmax(self.wavelet_weights, dim=0).detach().cpu(),
            "raw_wav": raw_wav.detach().cpu(),
            "x_wav_attended": x_wav_attended.detach().cpu(),
            "mother_wavelet": self.mother_wavelet.detach().cpu(),
            "mother_wavelet_norm": self._normalized_mother_wavelet().detach().cpu(),
        }

        return x_wav_attended, wav_att_weights
    
    
class CrossDomainIntegrator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.component_weights = nn.Parameter(torch.ones(3) / 3.0)
        self.temp = nn.Parameter(torch.tensor(1.0))
        self.residual_scaling = nn.Parameter(torch.tensor(0.5))

    def get_fusion_weights(self):
        w = torch.softmax(self.component_weights / self.temp, dim=0)
        return w.detach().cpu()

    def get_residual_weight(self):
        return float(torch.sigmoid(self.residual_scaling).detach().cpu().item())

    def get_temperature(self):
        return float(self.temp.detach().cpu().item())

    def forward(
        self,
        x_attended: torch.Tensor,
        x_guided: torch.Tensor,
        x_wav_attended: torch.Tensor,
        original: torch.Tensor,
    ) -> torch.Tensor:

        for name, tensor in {
            "x_attended": x_attended,
            "x_guided": x_guided,
            "x_wav_attended": x_wav_attended,
            "original": original,
        }.items():
            if tensor.ndim != 2:
                raise ValueError(f"{name} must have shape [B, L], got {tuple(tensor.shape)}")

        w = torch.softmax(self.component_weights / self.temp, dim=0)

        combined = (
            w[0] * x_attended +
            w[1] * x_guided +
            w[2] * x_wav_attended
        )

        enhanced = combined + torch.sigmoid(self.residual_scaling) * original

        return enhanced