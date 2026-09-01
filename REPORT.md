# DATA-266 HW1 Report

## 1. Autoregressive Models

An autoregressive model predicts the next value or token in a sequence using previously observed values or tokens. The model therefore represents a sequence as a series of conditional predictions, where each new prediction depends on the preceding context.

A simple real-world example is next-word prediction in language applications. Other examples include time-series forecasting, where previous measurements are used to predict a future measurement, and speech or sequence generation systems that predict subsequent elements from earlier context.

## 2. Diabetes Dataset Neural Network

### Objective

The objective was to preprocess the provided diabetes dataset, visualize its relationships and feature distributions, and compare a common feedforward neural network implementation in PyTorch and TensorFlow. The assignment requires the same preprocessing, split, and evaluation metric across the frameworks. fileciteturn2file2L135-L149

### Personal Configuration

- SID4 = 6359
- SEED = 6359
- HP_ID = 5
- HP_ID meaning = Schedule-long
- Baseline = [64, 32], learning rate 0.001, 30 epochs
- Modified = [64, 32], learning rate 0.001, 60 epochs

### Data Processing

The provided CSV has no header row. It was loaded with the eight input features and the binary `Outcome` label. The data was split into 70% training, 15% validation, and 15% test sets using the assigned seed. Standardization was fit on the training set and then applied to validation and test data to avoid data leakage.

The resulting split sizes were:

- Training: 531 samples
- Validation: 114 samples
- Test: 114 samples

### PyTorch Results

The baseline achieved a mean test accuracy of **0.7690 ± 0.0083** across the three training seeds. The HP_ID=5 model achieved **0.7749 ± 0.0149**.

The modified model therefore had a small improvement in mean accuracy, but the improvement was smaller than the modified model's measured variability. The longer training schedule did not provide strong evidence of a reliable improvement. fileciteturn2file8L496-L515

### TensorFlow Results

The baseline achieved **0.7749 ± 0.0149**, while the HP_ID=5 model achieved **0.7602 ± 0.0109**.

The TensorFlow result therefore moved in the opposite direction from PyTorch. The completed loss-curve analysis shows that training loss continued decreasing with additional epochs while validation loss plateaued and then drifted upward, especially between approximately epochs 30 and 60. This is consistent with overfitting. fileciteturn2file5L330-L332

### Neural-Network Conclusion

Doubling the epoch budget from 30 to 60 did not consistently improve generalization. PyTorch showed only a small mean improvement, while TensorFlow showed a decrease in mean test accuracy. The loss curves indicate that the additional training continued improving training fit without providing corresponding validation improvement. Therefore, for this experiment, the Schedule-long modification was not a reliable improvement and plausibly increased overfitting.

## 3. CUDA Matrix Multiplication

### Implementation

The CUDA implementation performs square matrix multiplication using a 16×16 thread block. Each thread computes exactly one element of the output matrix, and the grid is sized using ceiling division so that the full matrix is covered even when its dimensions are not an exact multiple of the block size. fileciteturn0file1L83-L91

For a block size of 16×16, each block contains 256 threads. The row and column of each output element are determined from `blockIdx`, `blockDim`, and `threadIdx`.

### CPU Comparison

The CPU baseline uses OpenMP to parallelize the outer row loop. The GPU path measures host-to-device transfer, kernel execution, and device-to-host transfer separately. A warm-up kernel launch is performed before the timed kernel to reduce first-launch overhead in the reported kernel measurement. fileciteturn0file1L249-L325

### Benchmark Results

| N | CPU (ms) | GPU kernel (ms) | H2D+D2H (ms) | GPU end-to-end (ms) | Speedup |
|---:|---:|---:|---:|---:|---:|
| 256 | 27.4259 | 0.1509 | 1.0478 | 1.1988 | 22.8787× |
| 1024 | 4910.4850 | 9.1950 | 5.5728 | 14.7678 | 332.5121× |
| 4096 | 693849.9563 | 326.8400 | 82.1530 | 408.9930 | 1696.4839× |

The GPU is substantially faster at all three tested sizes. The advantage becomes much larger as N increases because matrix multiplication has cubic computational growth while the GPU can execute many output-element calculations concurrently. fileciteturn0file1L454-L493

### Profiling

Nsight Systems was unavailable in the Colab runtime. Nsight Compute was available and used instead. The profiler successfully captured the CUDA kernel and produced an `.ncu-rep` report. The profiler-instrumented 1024 run reported approximately 1758.8 ms kernel time, but this is not directly comparable to the 9.195 ms unprofiled kernel timing because profiling instrumentation adds substantial overhead. The unprofiled values are therefore used for the benchmark table. fileciteturn1file4L205-L207

### Crossover Analysis

The GPU was already beneficial at N=256, the smallest tested size, with approximately 1.20 ms end-to-end GPU time versus 27.43 ms CPU time. The crossover is not at zero because GPU computation has fixed costs such as kernel launch and host/device data transfer. At N=256, the transfer component alone was approximately 1.05 ms, while the kernel itself took only about 0.15 ms. As N increases, the O(N³) computation becomes dominant and the GPU's parallelism produces increasingly large speedups. fileciteturn1file9L485-L487

## 4. Overall Conclusion

This homework demonstrated two different effects of parallel and extended computation. In the neural-network experiment, increasing the training budget did not reliably improve generalization and produced evidence of overfitting. In the CUDA experiment, GPU parallelism produced very large speedups over the CPU baseline, and the advantage increased dramatically with matrix size. The experiments also show why benchmark methodology matters: the CUDA profiler was useful for kernel-level analysis, but its instrumentation overhead meant that profiled and unprofiled timing measurements had to be interpreted separately.
