# METRICS.md: HW4.5 summary
_HW4.5 summary only. This is NOT an HW5 baseline._

Hardware: 2 x Tesla T4 (15 GB, PCIe via host bridge/PHB), Kaggle. Model: bert-base-uncased, tweet_eval/sentiment rows 0-4999 (SLICE 0, SEED 42), max_len 128, 4 epochs, lr 2e-5.

| Summary | Configuration | Value |
|---|---|---|
| Best peak memory | Gradient checkpointing, batch 32 | 2.24 GB (baseline 4.01 GB) |
| Best throughput | fp16 AMP + AdamW8bit, batch 32 | 161.9 samples/s (baseline 48.8) |
| Best wall-clock (4-epoch fine-tune, 3,500 rows) | fp16 AMP + AdamW8bit, batch 32 | 86.8 s (baseline 289.9 s), test acc 0.680 |
| Recommended full fine-tune | fp16 AMP + AdamW8bit, batch 32, 1 x T4 | 0.31 GPU-h for 4 epochs (0.16 GPU-h for 2) |

HW4 baseline used for HW4.5: peak memory 4.01 GB, training time 289 s (fp32, batch 32, 1 x T4).
