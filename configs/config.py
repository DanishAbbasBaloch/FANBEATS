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

# Configuration for solar wind speed forecasting experiments

LOOKBACK_LENGTH = 96

# Supported forecast horizons (hours): 24, 48, 72, 96
FORECAST_HORIZON = 24

NBEATS_MODE = "interpretable"

BATCH_SIZE = 128
LEARNING_RATE = 1e-3
MAX_STEPS = 4000

TIMESERIES_FREQUENCY = 24

# loss function used
LOSS_NAME = "HYBRID"

EVAL_WINDOW_STEP = 1

MODEL_PARAMS = {
    "interpretable": {
        "trend_blocks": 2,
        "trend_layers": 4,
        "trend_layer_size": 256,
        "degree_of_polynomial": 2,
        "seasonality_blocks": 2,
        "seasonality_layers": 4,
        "seasonality_layer_size": 1024,
        "num_of_harmonics": 4,
    },
}


OUTPUT_ROOT = "outputs"
TRAINED_MODELS_DIR = f"{OUTPUT_ROOT}/trained_models"
METRICS_DIR = f"{OUTPUT_ROOT}/metrics"
PREDICTIONS_DIR = f"{OUTPUT_ROOT}/predictions"
FIGURES_DIR = f"{OUTPUT_ROOT}/figures"
INTERPRETABILITY_DIR = f"{OUTPUT_ROOT}/interpretability"

TRAIN_FILE = "dataset/solar_wind_speed_train_df.csv"
VAL_FILE = "dataset/solar_wind_speed_val_df.csv"
TEST_FILE = "dataset/solar_wind_speed_test_df.csv"


def make_experiment_tag(mode: str, lookback: int, horizon: int, seed: int) -> str:
    """Build a compact experiment tag for filenames."""
    if seed == 32:
        return f"fanbeats_{mode}_lb{lookback}_h{horizon}"
        
    else: #changing the tag name based on the seed value to ensure reproducibility and clarity in experiment tracking
        return f"fanbeats_{mode}_lb{lookback}_h{horizon}_seed{seed}"
