# NMR Water Regression

This project focuses on analyzing and modeling Nuclear Magnetic Resonance data to predict fluid properties in porous media, specifically targeting petrophysical variables like irreducible water saturation (`Swirr_PHIX`).

Using a synthetic NMR dataset, the project implements a complete machine learning pipeline, ranging from advanced signal denoising using wavelets to modeling with Physics-Informed Neural Networks (PINNs).

## Project Structure

The project workflow is divided into two primary Jupyter Notebooks:

### 1. Data Preprocessing (`preprocessing.ipynb`)

This notebook handles data cleaning, signal processing, and dataset formatting:

- **Data Loading**: Ingests raw NMR data (`GulfCoast_RMN_Synthetic.csv`).
- **Signal Denoising**: Applies a Discrete Wavelet Transform (DWT) filter using Daubechies wavelets (`db6`, level 6, soft thresholding) across the entire dataset to remove noise from the NMR relaxation curves.
- **Feature Extraction**: Isolates the 500-point denoised signal features and separates the target petrophysical variables (`Swirr_PHIX`, `MBVI`, `MPHI`, `PHIX`).
- **Output**: The fully processed dataset is exported as `model_input.csv` for the modeling phase.

### 2. Modeling & Machine Learning (`modeling.ipynb`)

This notebook leverages the preprocessed data to train deep learning models using PyTorch:

- **Dimensionality Reduction**: Applies Principal Component Analysis (PCA) to reduce the dimensionality of the input data while retaining 99.9% of the variance. Data augmentation is also applied to the training set.
- **Multilayer Perceptron (MLP)**: Implements a baseline feedforward neural network to predict the target variable directly from the NMR signals.
- **Physics-Informed Neural Network (PINN)**: Implements an advanced, custom neural network architecture constrained by the physical equations of NMR $T_2$ relaxation.
  - **Physical Model**: Assumes a bi-exponential relaxation curve representing oil ($o$) and water ($w$) phases:
    $$
    f(t) = A_o e^{-t/T_{2,o}} + A_w e^{-t/T_{2,w}}
    $$
  - **Architecture**: The network outputs 4 distinct physical parameters: $A_o$, $A_w$, $T_{2,o}$, and $T_{2,w}$.
  - **Custom Loss Function**: The training error is composed of two synergistic parts:
    1. **Output Error**: Compares the physical fluid fraction ($A_o/(A_o+A_w)$) against the traditional target label $y$.
    2. **Fit Error**: Uses the network's 4 predicted parameters to mathematically reconstruct the curve $f(t)$, evaluating the reconstruction error against the true input curve $x$.

## Requirements

The project requires the following primary Python libraries:

- `numpy`, `pandas` (Data manipulation)
- `matplotlib`, `seaborn` (Data visualization)
- `PyWavelets` / `pywt` (DWT Denoising)
- `scikit-learn` (PCA and preprocessing)
- `PyTorch` (Deep learning and PINN implementation)
