# GPU-hour record (HW4.5)
Full fine-tune estimate = 45,615 rows × 4 epochs ÷ measured samples/s ÷ 3600
- Baseline fp32, 1×T4: 182,460 ÷ 48.3 = 3,778 s = 1.05 GPU-h
- fp16 + 8-bit, 1×T4: 182,460 ÷ 161.4 = 1,131 s = 0.31 GPU-h
- fp32 DDP 2×T4: 182,460 ÷ 100.2 = 1,821 s wall × 2 GPUs = 1.01 GPU-h
Actual GPU time used for this assignment: 2.53 hours (2 h 32 min of Kaggle GPU quota, accelerator GPU T4 x2, session 2-3 Nov 2026) (session ~21:30–23:50 PT, 2 Nov 2026, GPU T4 x2)
