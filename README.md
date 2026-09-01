# DATA-266 HW1 — Generative AI and LLM

## Overview

This repository contains the completed Homework 1 work for DATA-266, Fall 2026. The assignment includes:

1. A short explanation of autoregressive models and real-world examples.
2. Diabetes-dataset preprocessing, visualization, and feedforward neural-network experiments in PyTorch and TensorFlow.
3. A CUDA C matrix-multiplication implementation, CPU comparison, timing measurements, and Nsight Compute profiling.

The implementation and experimental work were completed independently. This documentation organizes the completed work and reported results into repository-ready Markdown files.

## Personal Parameters

| Parameter | Value |
|---|---:|
| SID4 | 6359 |
| SEED | 6359 |
| SLICE | 359 |
| HP_ID | 5 |
| HP_ID arm | Schedule-long |
| CLS_A | 9 |
| CLS_B | 5 |

For HW1, HP_ID=5 means the modified neural network uses the same architecture and learning rate as the baseline but trains for 60 epochs instead of 30. The assignment defines this as the Schedule-long configuration. fileciteturn2file4L205-L223

## Repository Contents

Recommended repository layout:

```text
data266-6359/
├── neural_networks.ipynb
├── cuda.ipynb
├── matmul.cu
├── METRICS.md
├── RUN_LOG.txt
├── AI_USE.md
├── REPORT.md
├── DOCUMENTATION.md
├── metrics_nn.csv
├── cuda_timing.csv
├── figures/
│   ├── correlation_matrix.png
│   ├── feature_distributions.png
│   ├── pytorch_loss_curves.png
│   └── tensorflow_loss_curves.png
└── README.md
```

The assignment specifically requires the executed notebooks, CUDA source, one document file, and the standing repository artifacts including `RUN_LOG.txt`, `METRICS.md`, and `AI_USE.md`. fileciteturn1file3L143-L152

## Reproducibility

The neural-network experiment uses a fixed 70/15/15 train/validation/test split with `random_state=SEED`. The same split is reused for all model comparisons, and the models are trained with seeds `SEED`, `SEED+1`, and `SEED+2`. fileciteturn1file0L19-L24

The CUDA benchmark uses the same personal seed for generating the input matrices. The CUDA kernel uses 16×16 thread blocks, with each thread computing one output element. fileciteturn0file1L83-L91

## Main Findings

- PyTorch baseline mean test accuracy: **0.7690**
- PyTorch HP_ID=5 mean test accuracy: **0.7749**
- TensorFlow baseline mean test accuracy: **0.7749**
- TensorFlow HP_ID=5 mean test accuracy: **0.7602**
- The longer 60-epoch schedule did not produce a consistent improvement and showed an overfitting pattern in the loss curves. fileciteturn2file5L330-L332
- GPU end-to-end execution was faster than the CPU baseline for all tested matrix sizes: 256, 1024, and 4096. fileciteturn1file9L485-L487

## CUDA Environment

The benchmark ran on a Google Colab GPU environment using a Tesla T4 with CUDA 13.0. Nsight Systems (`nsys`) was unavailable, while Nsight Compute (`ncu`) was available and used for profiling. fileciteturn0file1L22-L40 fileciteturn1file4L222-L254
