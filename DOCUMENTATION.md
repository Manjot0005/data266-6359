# HW1 Documentation / Reproducibility Guide

## Purpose

This document explains how the completed HW1 artifacts fit together and how another person can understand or reproduce the reported work.

## 1. Neural Network Notebook

File: `neural_networks.ipynb`

The notebook should contain:

1. Personal parameters and seed setup.
2. Dataset loading and preprocessing.
3. Correlation matrix.
4. Feature-distribution plots.
5. Fixed 70/15/15 train/validation/test split.
6. Standardization using training data only.
7. PyTorch baseline and HP_ID=5 model.
8. Three-seed PyTorch experiment.
9. PyTorch training/validation loss curves.
10. TensorFlow baseline and HP_ID=5 model.
11. Three-seed TensorFlow experiment.
12. TensorFlow training/validation loss curves.
13. Final comparison table.

The completed experiment used `SEED=6359`, with training seeds 6359, 6360, and 6361. The split itself remained fixed across model comparisons. fileciteturn1file0L19-L24

## 2. CUDA Notebook

File: `cuda.ipynb`

The notebook should contain:

1. GPU availability check.
2. CUDA source creation.
3. Compilation command.
4. Matrix multiplication execution for N=256, 1024, and 4096.
5. Timing results.
6. CPU/GPU correctness comparison.
7. Profiler availability checks.
8. Nsight Compute profiling output.
9. Crossover analysis.

The assignment explicitly asks for CPU time, GPU kernel time, H2D+D2H transfer time, and speedup for the three matrix sizes. fileciteturn1file0L21-L31

## 3. CUDA Source

File: `matmul.cu`

The kernel uses:

```text
BLOCK_SIZE = 16

threadsPerBlock = (16, 16)
                = 256 threads/block

numBlocks = (ceil(N/16), ceil(N/16))
```

Each thread maps to one `(row, col)` location in C and performs the dot product over the K dimension. The kernel includes bounds checks so that matrices whose dimensions are not multiples of 16 remain valid. fileciteturn0file1L107-L145

The program also includes:

- CPU matrix multiplication
- deterministic random input generation
- CUDA error checking
- host/device memory allocation
- H2D and D2H timing
- kernel timing with CUDA events
- CPU/GPU numerical validation
- CSV-formatted benchmark output

## 4. Benchmark Methodology

The GPU kernel is warmed up before the timed launch. CUDA events are used for kernel and transfer timing. End-to-end GPU time is defined as kernel time plus H2D+D2H transfer time, and speedup is calculated as:

```text
speedup = CPU time / GPU end-to-end time
```

This definition matches the reported benchmark values. fileciteturn0file1L307-L359

## 5. Profiling Methodology

`nsys` was checked first but was not installed. `ncu` was then used for kernel-level profiling.

Important: profiler output is not used as the benchmark timing table because profiling adds instrumentation overhead. The completed run explicitly observed approximately 1.76 seconds of profiled kernel time versus approximately 9.2 milliseconds in the normal N=1024 benchmark. fileciteturn1file4L205-L207

## 6. Expected Repository Artifacts

The course standing requirements call for:

- executed notebook(s) with outputs intact
- `RUN_LOG.txt`
- `METRICS.md`
- `AI_USE.md`
- model checkpoints when applicable
- `figures/` only when a failure gallery is explicitly required

The assignment submission also requires `neural_networks.ipynb`, `cuda.ipynb`, the CUDA `.cu` source, and one document file. fileciteturn1file3L143-L152 fileciteturn2file3L182-L188

## 7. Suggested Git Workflow

Because the course states that incremental commits matter, avoid submitting a repository with only one last-minute commit. A reasonable history is:

```text
git add neural_networks.ipynb
git commit -m "Complete neural network experiments"

git add cuda.ipynb matmul.cu
git commit -m "Complete CUDA matrix multiplication and profiling"

git add METRICS.md RUN_LOG.txt AI_USE.md README.md REPORT.md DOCUMENTATION.md
git commit -m "Add HW1 documentation and metrics"

git tag hw1
```

The assignment states that a repository whose entire history is a single near-deadline commit can be capped on the implementation portion. fileciteturn1file3L154-L154
