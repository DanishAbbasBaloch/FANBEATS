# Dataset Preparation for FANBEATS

This directory contains the processed dataset used for training and evaluating the FANBEATS model, along with a preprocessing notebook to reproduce the data pipeline.

---

## 1. Data Source

The dataset is derived from the publicly available **NASA OMNI solar wind database**:

https://omniweb.gsfc.nasa.gov/

Specifically, hourly solar wind bulk speed data (km/s) from **2011 to 2017** are used in this study.

---

## 2. Provided Files (Ready to Use)

For convenience and reproducibility, the following processed datasets are already provided:

* `solar_wind_speed_train_df.csv`
* `solar_wind_speed_val_df.csv`
* `solar_wind_speed_test_df.csv`

These files:

* are **chronologically split**
* include **missing-value imputation**
* are **ready for direct use in FANBEATS**

Users can directly use these files without additional preprocessing.
These files are derived from the NASA OMNI dataset and are provided for research and reproducibility purposes.

---

## 3. Reproducing the Dataset (Optional)

To reproduce the dataset from raw OMNI data:

### Step 1: Download Raw Data

* Go to: https://omniweb.gsfc.nasa.gov/ow.html
* Select **hourly resolution**
* Select **solar wind bulk speed variable**
* Download data for **2011–2017**
* Save the file as:

```text
org_solar_wind_speed_data.csv
```

Place this file inside the `dataset/` directory.

---

### Step 2: Run Preprocessing Notebook

Open and run:

```text
data_preprocessing.ipynb
```

This notebook performs:

* chronological train/validation/test split
* missing value imputation:

  * linear interpolation
  * forward/backward filling (boundary cases)

---

### Step 3: Generated Files

After execution, the following files will be created:

* `solar_wind_speed_train_df.csv`
* `solar_wind_speed_val_df.csv`
* `solar_wind_speed_test_df.csv`

---

## 4. Dataset Splits

The data are divided chronologically to preserve temporal structure:

| Subset     | Period    | Samples |
| ---------- | --------- | ------- |
| Training   | 2011–2015 | 43,824  |
| Validation | 2016      | 8,784   |
| Test       | 2017      | 8,760   |

For final training, training and validation sets may be combined.

---

## 5. Preprocessing Summary

* Missing values are handled using:

  * linear interpolation
  * forward/backward filling
* No smoothing or outlier removal is applied
* Data normalisation is **not performed at this stage**
  (handled during model training)

---

## 6. Notes

* The dataset is derived from the NASA OMNI database.
* The provided CSV files correspond to the processed dataset used in this study.
* All preprocessing steps are implemented in the notebook for full reproducibility.

---
