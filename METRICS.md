# HW1 Metrics

## Neural Network Experiments

### Configuration

| Model | Framework | Hidden layers | Learning rate | Epochs |
|---|---|---|---:|---:|
| Baseline | PyTorch | [64, 32] | 0.001 | 30 |
| HP_ID=5 | PyTorch | [64, 32] | 0.001 | 60 |
| Baseline | TensorFlow | [64, 32] | 0.001 | 30 |
| HP_ID=5 | TensorFlow | [64, 32] | 0.001 | 60 |

HP_ID=5 is the assignment's Schedule-long configuration. fileciteturn2file1L73-L89

### Test Accuracy Across Three Seeds

| Framework | Model | Seed 6359 | Seed 6360 | Seed 6361 | Mean | Std. dev. |
|---|---|---:|---:|---:|---:|---:|
| PyTorch | Baseline | 0.7632 | 0.7807 | 0.7632 | **0.7690** | **0.0083** |
| PyTorch | HP_ID=5 | 0.7807 | 0.7895 | 0.7544 | **0.7749** | **0.0149** |
| TensorFlow | Baseline | 0.7807 | 0.7895 | 0.7544 | **0.7749** | **0.0149** |
| TensorFlow | HP_ID=5 | 0.7719 | 0.7632 | 0.7456 | **0.7602** | **0.0109** |

The assignment requires three training seeds while keeping the data split fixed. fileciteturn1file0L19-L24

### Neural-Network Interpretation

The HP_ID=5 modification changes only the training budget from 30 to 60 epochs. In PyTorch, the mean accuracy increases from 0.7690 to 0.7749, a difference of 0.0059, which is smaller than the modified model's standard deviation of 0.0149. In TensorFlow, the mean accuracy decreases from 0.7749 to 0.7602, a difference of -0.0147. The completed loss-curve interpretation identifies this as an overfitting pattern: training loss continues to decrease while validation loss plateaus and can rise, especially in TensorFlow. fileciteturn2file5L330-L332

## CUDA Matrix-Multiplication Metrics

| Matrix size | CPU (ms) | GPU kernel (ms) | H2D+D2H (ms) | GPU end-to-end (ms) | Speedup |
|---:|---:|---:|---:|---:|---:|
| 256 | 27.4259 | 0.1509 | 1.0478 | 1.1988 | **22.8787×** |
| 1024 | 4910.4850 | 9.1950 | 5.5728 | 14.7678 | **332.5121×** |
| 4096 | 693849.9563 | 326.8400 | 82.1530 | 408.9930 | **1696.4839×** |

The recorded CUDA results include the CPU baseline, isolated GPU kernel time, combined host/device transfer time, end-to-end GPU time, and numerical difference between CPU and GPU outputs. fileciteturn0file1L454-L493

### CUDA Validation

| Matrix size | Max absolute CPU/GPU difference |
|---:|---:|
| 256 | 1.525879e-05 |
| 1024 | 9.155273e-05 |
| 4096 | 3.662109e-04 |

The small floating-point differences are consistent with evaluating the same matrix multiplication using CPU and GPU floating-point arithmetic.

## Profiler

**Profiler used:** NVIDIA Nsight Compute (`ncu`).

`nsys` was attempted but was not installed in the Colab environment. `ncu` was available and successfully produced a profiler report. The profiled 1024 run reported a much larger kernel time (~1758.8 ms) because profiler instrumentation adds substantial overhead. Therefore, the unprofiled timing table above is the correct source for benchmark metrics; the profiled run is used for kernel-level analysis only. fileciteturn1file4L205-L207 fileciteturn1file4L246-L271

## GPU Crossover

The GPU was already beneficial at the smallest tested size, N=256: approximately 1.20 ms end-to-end on GPU versus 27.43 ms on CPU. The crossover is not at size zero because GPU execution has fixed overheads such as kernel launch and host/device data transfer; at N=256, transfer alone was about 1.05 ms. As matrix size grows, computation scales approximately as O(N³), while data transfer scales approximately as O(N²), so the GPU advantage becomes increasingly large. fileciteturn1file9L485-L487
