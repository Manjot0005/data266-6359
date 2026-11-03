# HW4.a + HW4.5 — BERT fine-tune, vLLM serving, memory and throughput engineering

Every script writes raw measurements to `results/<GPU name>/*.json`. `report.py` turns those into
`REPORT_TABLES.md`, `METRICS.md` and `plots/`. No number is hard-coded, so anything in your report
comes from a run you actually did.

## Setup (Kaggle / Colab / lab box)
```bash
pip install -U transformers datasets scikit-learn bitsandbytes vllm matplotlib pandas
```
* **Kaggle "GPU T4 x2"** gives you the two-GPU Part C pairing: homogeneous, single node, PCIe
  (check with `nvidia-smi topo -m`; the script saves it). Use the pairing your instructor designates for your slot.
* **Two different cards for Part A/D.** Run `run_all.sh` once on each card (for example the lab 5090 and 4090,
  or Kaggle P100 vs T4) and copy both `results/<card>/` folders into one `results/` before running `report.py`.
* T4 and P100 have no bf16 support, so AMP falls back to fp16 with a GradScaler and vLLM runs with `--dtype half`.
  Both scripts pick this automatically.

## Data: SLICE / SEED
`--csv file --text-col <col> --label-col <col> --slice $SLICE --seed $SEED`. For a CSV with no header
(Sentiment140), add `--no-csv-header --label-col 0 --text-col 5`. Leave out `--csv` to use HF `tweet_eval/sentiment`.
The script uses rows `[SLICE, min(SLICE+5000, N)-1]`, or all the remaining rows if fewer than 5,000 are left.
It records the exact range in every JSON. The 70/15/15 split comes from a permutation seeded with SEED,
and SEED also seeds model initialisation and the data order.
**Sentiment140 is sorted by label.** If your slice contains only one class, the script stops with an error
instead of training a degenerate model.

## Run order (one card)
```bash
SLICE=<yours> SEED=<yours> CSV=twitter.csv TEXT_COL=text LABEL_COL=label bash run_all.sh
```
| Step | Script | Covers |
|---|---|---|
| 1 | `hw4a_train.py` | HW4.a: per-epoch losses/accuracy, epoch where validation stopped improving and the time spent past it, confusion matrix, P/R, 5 errors from the worst class, peak memory and training time, save/reload/inference |
| 2 | `serve_bench.py --mode hf / vllm` | Table HW4.1 and Part D: tok/s, p50/p95 at batch 1/8/32 vs the HF batch-1 pipeline |
| 3 | `partA_batch_scaling.py --full-epoch` | Part A: batch doubling until OOM |
| 4 | `partB_memory_techniques.py` | Part B: AMP, checkpointing, accumulation, 8-bit Adam, each alone, then the combo |
| 5 | `serve_bench.py --mode kv` | Part D: KV memory vs context length, longest context at batch 8 |
| 6 | `torchrun … partC_ddp.py` | Part C: scaling efficiency, comm overhead per step, all-reduce bandwidth |
| 7 | `report.py` | All tables, the plot, Part E GPU-hours, METRICS.md |

After step 1, open `results/<card>/hw4a_train.json` and fill in the five `"diagnosis"` lines by hand
(sarcasm, negation, label noise, neutral-but-emotive wording, and so on). Then re-run `report.py`.

## How each number is measured (so you can defend it)
* **Throughput (samples/s):** training-only wall time with `cuda.synchronize()`. Warm-up steps are excluded.
  Inputs are padded to a fixed `max_len=128`, so memory does not depend on the batch contents.
* **Peak memory:** `torch.cuda.max_memory_allocated` (tensors) and `max_memory_reserved` (what `nvidia-smi`-style
  OOMs care about). Report allocated; mention reserved when explaining the OOM point.
* **Part B comparability:** each technique gets an OOM probe for max batch, plus throughput and memory measured at the
  *same* HW4 reference batch (32 by default), plus a full fine-tune for test accuracy. The ranking table divides
  GB saved by % throughput lost. Techniques that speed things up (AMP) rank at the top.
* **Gradient accumulation:** physical 8 × 32 steps = effective 256. It is compared with physical batch 256 (if that fits)
  at the *same number of optimiser steps*, giving wall-clock and accuracy for both.
* **Serving:** greedy decoding, the same number of generated tokens per request (`ignore_eos`/`min_new_tokens`),
  20 timed batches. A request's latency is the wall time of the batch it ran in.
* **DDP comm overhead:** step time with all-reduce minus step time under `no_sync()`. This is the *exposed*
  communication cost, after overlap with backward. The script also runs a standalone NCCL all-reduce on a
  gradient-sized buffer (≈438 MB fp32 for BERT-base) to get algbw/busbw. In a heterogeneous pair, the per-rank
  compute times show how long the fast card waits for the slow one.
* **KV cache:** the analytic bytes/token is `2·layers·kv_heads·head_dim·2 B`. It is cross-checked against the
  `past_key_values` tensors that HF actually allocates at batch 8. For vLLM, KV pool tokens ÷ 8 gives the longest
  batch-8 context, and a real 8-request generation at that length verifies it.

## Short-answer guidance (fill in with your numbers)
* **Latency vs throughput crossover (HW4.1):** `report.py` prints the first vLLM batch whose p50 exceeds the HF batch-1
  p50 while tok/s is still rising. For an interactive app, per-request (p95) latency is the number that matters.
  Total throughput is a cost/capacity metric for offline or bulk serving.
* **Max-batch ratio ≠ VRAM ratio (Part A.3):** weights, gradients, Adam moments (≈16 B/param ≈ 1.75 GB for BERT-base)
  and the CUDA context are a fixed cost paid before any activations, so only the leftover VRAM scales with batch.
  Power-of-two probing also rounds the ratio, and allocator fragmentation differs between cards.
* **Validation stopped improving:** `best_val_epoch` and `extra_time_past_best_s` in the JSON. The cost of training
  past that point is that extra time plus any rise in validation loss or drop in accuracy.
* **"Which first / which not to bother with on 32 GB":** take this from your ranking table. Usually AMP comes first
  because it saves memory *and* runs faster on tensor cores. At BERT-base scale, the 8-bit optimiser saves only ~1 GB of
  optimiser state, and checkpointing costs ~20–35% throughput for memory you don't need on 32 GB. Confirm both against
  your own numbers rather than quoting these.
* **Part E:** `report.py` computes GPU-hours = full-dataset train rows × epochs ÷ measured samples/s ÷ 3600.
  Pass `--full-dataset-rows` for your dataset (1,600,000 for Sentiment140) and `--epochs`.
