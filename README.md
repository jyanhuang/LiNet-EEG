# LiNet-EEG

This repository contains the PyTorch implementation of LiNet-EEG, a lightweight convolutional neural network for EEG-based driver drowsiness recognition.

## Network Architecture

LiNet-EEG consists of:

- Pointwise convolution
- Depthwise temporal convolution
- ReLU activation and batch normalization
- Squeeze-and-Excitation (SE) attention
- Temporal average pooling
- Fully connected classification layer

The input EEG sample has a shape of `(1, 30, 384)`. The model contains 4,354 trainable parameters. After layer-wise unstructured pruning, 3,762 nonzero parameters remain.

## Repository Structure

```text
LiNet-EEG/
├── README.md
├── requirements.txt
├── train.py
├── model.py
├── EDDA.py
├── utils.py
└── EEGdataset/
    └── dataset.mat
