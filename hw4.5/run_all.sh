#!/usr/bin/env bash
# Run on ONE card. Repeat on the second card (Part A/D need two cards), then run report.py
# over the merged results/ folder.   Usage:  SLICE=12000 SEED=7 CSV=twitter.csv bash run_all.sh
set -euo pipefail
: "${SLICE:?set SLICE}"; : "${SEED:?set SEED}"
CSV_ARGS=${CSV_ARGS:-"--csv ${CSV:-} --text-col ${TEXT_COL:-text} --label-col ${LABEL_COL:-label}"}
[ -z "${CSV:-}" ] && CSV_ARGS=""            # empty -> HF tweet_eval/sentiment
COMMON="$CSV_ARGS --slice $SLICE --seed $SEED"
MODEL=${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}
# T4/P100/V100 have no bf16 -> vLLM needs half
DTYPE=$(python -c "import torch;print('auto' if torch.cuda.is_bf16_supported() else 'half')")

python hw4a_train.py $COMMON                                   # HW4.a 1
python serve_bench.py --mode hf   --model $MODEL --dtype $DTYPE # HW4.a 2 / Part D baseline
python serve_bench.py --mode vllm --model $MODEL --dtype $DTYPE
python partA_batch_scaling.py $COMMON --full-epoch             # Part A
python partB_memory_techniques.py $COMMON                      # Part B (all 4 + combo)
python serve_bench.py --mode kv   --model $MODEL --dtype $DTYPE # Part D KV cache
NG=$(python -c "import torch;print(torch.cuda.device_count())")
torchrun --nproc_per_node=1 partC_ddp.py $COMMON                # Part C single-GPU baseline
[ "$NG" -ge 2 ] && torchrun --nproc_per_node=2 partC_ddp.py $COMMON || echo "only $NG GPU: skip 2-GPU DDP"
python report.py --results results
