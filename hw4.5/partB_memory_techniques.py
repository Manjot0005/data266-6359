"""HW4.5 Part B: memory techniques, one at a time, then the best combination.

For each technique (applied ALONE on top of the HW4 baseline):
  * max batch      -- power-of-two OOM probe
  * throughput     -- samples/s at the HW4 reference batch (--batch-size, default 32)
  * peak memory    -- at the same reference batch (so memory saved is apples-to-apples)
  * eval accuracy  -- full HW4 fine-tune (same epochs, seed, split), test accuracy
Gradient accumulation instead reports: effective batch, wall-clock to the same number of optimiser
steps as a run that uses that batch physically (if it fits), and accuracy.

  python partB_memory_techniques.py --csv ... --slice $SLICE --seed $SEED [--only amp ckpt ...]
  python partB_memory_techniques.py ... --only combo --combo amp=bf16 optim=adamw8bit
"""
from dataclasses import asdict
import torch
from common import (Cfg, build_datasets, probe_max_batch, train_and_eval, save_json, card_slug,
                    gpu_name, free_cuda, is_oom)


def extra(p):
    p.add_argument("--only", nargs="*", default=["baseline", "amp", "ckpt", "accum", "optim8", "combo"])
    p.add_argument("--amp-dtype", default="auto")          # auto -> bf16 if supported else fp16
    p.add_argument("--accum-physical", type=int, default=8)
    p.add_argument("--accum-effective", type=int, default=256)
    p.add_argument("--optim8", default="adamw8bit")        # or adafactor if bitsandbytes unavailable
    p.add_argument("--combo", nargs="*", default=["amp=auto", "optim=adamw8bit"])
    p.add_argument("--probe-steps", type=int, default=10)
    p.add_argument("--skip-probe", action="store_true")


def variant(base: Cfg, **kw):
    return Cfg(**{**asdict(base), **kw})


def measure(name, cfg, splits, names, a):
    print(f"\n===== {name} =====")
    rec = {"technique": name, "cfg": asdict(cfg)}
    if not a.skip_probe:
        rows = probe_max_batch(cfg, splits, names, start=8, steps=a.probe_steps)
        ok = [r for r in rows if r["ok"]]
        rec["probe"] = rows; rec["max_batch"] = ok[-1]["batch_size"] if ok else None
        if cfg.accum > 1 and rec["max_batch"]:
            rec["max_effective_batch"] = rec["max_batch"] * cfg.accum
    # throughput + memory at the reference batch
    t = train_and_eval(cfg, splits, names, do_eval=False, timed_steps=30, quiet=True)
    rec.update(ref_batch=cfg.batch_size, samples_per_s=t["samples_per_s"],
               peak_mem_alloc_gb=t["peak_mem_alloc_gb"], peak_mem_reserved_gb=t["peak_mem_reserved_gb"])
    del t; free_cuda()
    # full fine-tune for accuracy
    f = train_and_eval(cfg, splits, names)
    f.pop("_model"); f.pop("_test_preds"); f.pop("_test_gold")
    rec.update(test_acc=f["test_acc"], train_time_s=f["train_time_s"], history=f["history"],
               full_run_samples_per_s=f["samples_per_s"], full_run_peak_mem_gb=f["peak_mem_alloc_gb"],
               opt_steps=f["opt_steps"])
    del f; free_cuda()
    print({k: v for k, v in rec.items() if k not in ("probe", "history", "cfg")})
    return rec


def main():
    base, a = Cfg.from_cli(extra)
    amp = a.amp_dtype if a.amp_dtype != "auto" else ("bf16" if torch.cuda.is_bf16_supported() else "fp16")
    splits, _, names, rr, _ = build_datasets(base)
    out = {"gpu": gpu_name(), "row_range": rr, "results": {}}
    path = f"{base.out}/{card_slug()}/partB.json"

    def run(key, name, cfg):
        out["results"][key] = measure(name, cfg, splits, names, a); save_json(out, path)

    if "baseline" in a.only: run("baseline", "Baseline (HW4 settings)", base)
    if "amp" in a.only:      run("amp", f"Mixed precision ({amp})", variant(base, amp=amp, tag="amp"))
    if "ckpt" in a.only:     run("ckpt", "Gradient checkpointing", variant(base, grad_ckpt=True, tag="ckpt"))
    if "optim8" in a.only:   run("optim8", f"8-bit optimiser ({a.optim8})", variant(base, optim=a.optim8, tag="optim8"))

    if "accum" in a.only:
        k = a.accum_effective // a.accum_physical
        n_train = len(splits["train"])
        steps = max(1, (n_train // a.accum_effective)) * base.epochs     # same optimiser steps for both
        acc = variant(base, batch_size=a.accum_physical, accum=k, max_opt_steps=steps, tag="accum")
        rec = measure(f"Gradient accumulation ({a.accum_physical}x{k}={a.accum_effective})", acc, splits, names, a)
        rec["effective_batch"] = a.accum_effective
        # reference: the same effective batch done physically, same number of optimiser steps
        phys = variant(base, batch_size=a.accum_effective, max_opt_steps=steps, tag="phys_large")
        try:
            f = train_and_eval(phys, splits, names)
            rec["physical_large_batch"] = {"batch": a.accum_effective, "train_time_s": f["train_time_s"],
                                           "test_acc": f["test_acc"], "peak_mem_alloc_gb": f["peak_mem_alloc_gb"],
                                           "opt_steps": f["opt_steps"]}
            del f
        except Exception as e:
            if not is_oom(e): raise
            rec["physical_large_batch"] = {"batch": a.accum_effective, "error": "OOM -- accumulation is the only way"}
        free_cuda()
        out["results"]["accum"] = rec; save_json(out, path)

    if "combo" in a.only:
        kw = {}
        for kv in a.combo:
            k, v = kv.split("=")
            if k == "amp" and v == "auto": v = amp
            if k in ("grad_ckpt",): v = v.lower() in ("1", "true", "yes")
            elif k in ("accum", "batch_size", "max_opt_steps"): v = int(v)
            kw[k] = v
        run("combo", "Best combination (" + ", ".join(a.combo) + ")", variant(base, tag="combo", **kw))


if __name__ == "__main__":
    main()
