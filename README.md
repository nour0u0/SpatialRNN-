# Patch-Based Image Reconstruction with Hidden State MLP SPATIAL RNN 

This repository contains a PyTorch implementation of a patch-based image reconstruction model using overlapping patches with a hidden state MLP architecture.

## Overview

- Dataset: CIFAR-10 (downloaded automatically)
- Model: MLP that predicts each patch and updates a hidden state sequentially
- Features:
  - Overlapping patch extraction and reconstruction with averaging
  - Training with MSE loss on patch reconstruction
  - Per-batch training and validation loss tracking and plotting
  - Visual comparison of original vs reconstructed images after training

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- tqdm
- numpy
- matplotlib

## Installation

Install required packages with:

```bash
pip install -r requirements.txt
