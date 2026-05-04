# FANBEATS: Frequency and Attention-augmented Neural Basis Expansion Analysis for interpretable Time Series forecasting

FANBEATS is an interpretable deep learning framework for **univariate solar wind speed forecasting**, designed to capture both long-range periodic structures and short-term temporal dynamics.

The model integrates:

* **Spectral Filtering and Attention Module (SFAM)** for frequency-domain representation
* **Multiscale Wavelet Attention Module (MWAM)** for temporal feature extraction
* **Cross-Domain Integrator (CDI)** for adaptive fusion
* A **modified interpretable N-BEATS backbone** for structured forecasting

---

## Overview

Accurate forecasting of solar wind speed is essential for space weather applications. However, existing approaches often rely on complex multimodal inputs or operate as black-box models, limiting interpretability and practical deployment.

FANBEATS addresses these limitations by integrating:

* frequency-domain filtering for global periodic behaviour
* multiscale temporal modelling for localised dynamics
* explicit trend–seasonality decomposition for interpretability

The framework operates on **univariate data**, avoiding dependence on auxiliary inputs while maintaining strong predictive performance across multiple forecasting horizons.

---

## Key Features

* ✔ Univariate solar wind speed forecasting
* ✔ Multi-horizon prediction (24, 48, 72, 96 hours)
* ✔ Joint modelling of spectral and temporal dynamics
* ✔ Adaptive fusion of complementary representations
* ✔ Multi-level interpretability:

  * spectral responses (SFAM)
  * multiscale temporal patterns (MWAM)
  * temporal attention mechanisms
  * trend–seasonality decomposition
* ✔ Event-based evaluation capability (HSS detection)

---

## Repository Structure

```text
FANBEATS/
│
├── configs/
│   └── config.py
│
├── dataset/
│   ├── README.md
│   ├── data_preprocessing.ipynb
│   ├── solar_wind_speed_train_df.csv
│   ├── solar_wind_speed_val_df.csv
│   └── solar_wind_speed_test_df.csv
│
├── factories/
│   └── nbeats_factories_modified.py
|
├── models/
│   ├── fanbeats_model.py
│   ├── fanbeats_modules.py
│   └── nbeats_modified.py
│
├── notebooks/
│   ├── fanbeats_main.ipynb
│   └── fanbeats_interpretability.ipynb
|
├── outputs/
|
├── third_party/
|   └── nbeats_license_notice.txt
|
├── trainers/
│   └── trainer.py
│
├── utils/
│   ├── ops.py
│   ├── sampler.py
│   └── solar_wind_dataset.py
│
├── LICENSE
├── README.md
├── requirements.txt
```

---

## Dataset

The model uses hourly solar wind speed data (2011–2017) derived from the NASA OMNI database.

Processed datasets are provided for direct use.
For full details and reproducibility:

```
dataset/README.md
```

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Train and Evaluate Model

Run:

```bash
notebooks/fanbeats_main.ipynb
```

This performs:

* data loading
* data normalisation
* model training
* multi-horizon evaluation

---

### 2. Interpretability Analysis

Run:

```bash
notebooks/fanbeats_interpretability.ipynb
```

This generates:

* spectral analysis (SFAM)
* multiscale analysis (MWAM)
* temporal attention weights
* trend–seasonality decomposition

---

## Outputs

Results are saved in:

```
outputs/
```

Including:

* predictions
* evaluation metrics
* trained models
* interpretability visualisations

---

## License

This repository is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)**.

It includes original FANBEATS contributions and modified components from N-BEATS.

See:

```
LICENSE
third_party/nbeats_license_notice.txt
```

---

## Citation

(To be updated after publication)

---

## Acknowledgements

This work builds upon the N-BEATS framework:

B. N. Oreshkin, D. Carpov, N. Chapados, and Y. Bengio,
“N-BEATS: Neural basis expansion analysis for interpretable time series forecasting,”
Proc. International Conference on Learning Representations (ICLR), 2020.

Original implementation: https://github.com/ServiceNow/N-BEATS

---
