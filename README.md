# Advanced Remaining Useful Life (RUL) Prediction Pipeline for NASA Turbofan Jet Engines

A complete, state-of-the-art deep learning and time-series engineering pipeline developed to predict the Remaining Useful Life (RUL) of aircraft jet engines using the **NASA C-MAPSS (Turbofan Engine Degradation Simulation) Dataset** across **FD001** and **FD002** operational profiles. 

This project covers the end-to-end predictive maintenance workflow: from raw simulation readings, robust multi-strategy feature extraction and selection, up to advanced sequential hybrid architectures (Conv1D + LSTM/GRU) and Autoencoder latent space modeling optimized via automated hyperparameter tuning.

## Project Overview

Predicting Remaining Useful Life (RUL) is a key part of smart machine maintenance. It helps factories avoid unexpected breakdowns and makes industrial equipment last much longer. This repository provides an organized, ready-to-use system that cleans the signals and trains accurate neural networks to figure out exactly when a machine will fail.

By comparing a simple setup (**FD001** with only one operating condition) to a complex real-world situation (**FD002** with six changing conditions), this project proves that mixing time-series convolutions (Conv1D) with recurrent layers (LSTM/GRU) gives steady and reliable results when tracking how machines wear out over time.

## Features

### 1. Data Cleaning & Target Engineering
* Automates missing value treatment.
* Implements variance-threshold checks to eliminate constant/near-constant sensors.
* Generates true target Remaining Useful Life (RUL) labels based on the maximum cycle lifetime achieved by each distinct unit.

### 2. Time-Series Processing
* Preprocesses flat multi-variate readings into overlapping temporal matrices via a sliding-window approach to accurately feed recurrent neural networks.

### 3. Multi-Strategy Feature Selection
Evaluates and narrows down features using multiple mathematical paradigms:
* **Filter Methods**: Correlation matrices and Mutual Information Regression coefficients.
* **Wrapper Methods**: Sequential Feature Selection (SFS) and Recursive Feature Elimination (RFECV) using TimeSeriesSplit cross-validation.
* **Embedded Methods**: Feature importance metrics derived from Random Forest Regressors and L1-penalized LASSO CV models.
* **Dimensionality Reduction**: Principal Component Analysis (PCA) along with PCA-weighted reconstruction subsets to compress complex operating states.

### 4. Optimization & Evaluation
* Leverages **GridSearchCV** combined with `KerasRegressor` and specialized `TimeSeriesSplit` cross-validation for rigorous grid tuning.
* Integrates **KerasTuner (Hyperband algorithm)** to dynamically find optimal network architectures efficiently.
* Rigorously evaluates RUL forecasting capabilities across multiple distinct feature domains (Raw features vs. Selection-subset features vs. Autoencoder latent dimensions) utilizing physical-cycle Root Mean Squared Error (RMSE) metrics.

## Models & Architectures Implemented

The pipeline moves past traditional regression to deploy specialized, state-of-the-art Deep Learning topologies:

* **Hybrid Conv1D + LSTM Regressor**
* **Hybrid Conv1D + GRU Regressor**
* **Autoencoder (AE) Representation Regressor**: Trains a sequence-to-sequence deep Autoencoder framework tasked with reconstructing healthy and degrading engine sequences down into compressed latent bottlenecks. The extracted **Encoder** head is then connected to a deep regression layer and evaluated via two distinct methodologies:
  * **Non-Trainable (Frozen Weights)**: The encoder behaves as a static, robust non-linear feature extractor; only the regression layers adapt to predict RUL.
  * **Trainable (Fine-Tuned)**: The entire encoder-to-regressor network is unfrozen, letting error gradients reshape representation filters directly for specialized RUL targeting.

## Dataset Profiles

The repository is built around the **NASA C-MAPSS Dataset**

