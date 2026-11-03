"""HW4.5 Part A: scale batch size by powers of two until OOM, on the HW4.a settings unchanged.

Run once per card:  python partA_batch_scaling.py --csv ... --slice $SLICE --seed $SEED
Per batch: samples/s (timed after warm-up), peak memory (allocated + reserved), time per epoch.
--full-epoch also times one real epoch per batch size instead of extrapolating from --steps.
"""
from dataclasses import asdict
from common import Cfg, build_datasets, probe_max_batch, train_and_eval, save_json, card_slug, gpu_name, free_cuda, is_oom
import torch


def extra(p):
    p.add_argument("--start-bs", type=int, default=8)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--full-epoch", action="store_true")


def main():
    cfg, a = Cfg.from_cli(extra)
    splits, _, names, rr, _ = build_datasets(cfg)
    rows = probe_max_batch(cfg, splits, names, start=a.start_bs, steps=a.steps)
    if a.full_epoch:
        for r in rows:
            if not r["ok"]:
                continue
            c = Cfg(**{**asdict(cfg), "batch_size": r["batch_size"], "epochs": 1})
            try:
                res = train_and_eval(c, splits, names, do_eval=False, quiet=True)
                r["measured_epoch_time_s"] = res["history"][0]["epoch_time_s"]
                del res
            except Exception as e:
                if not is_oom(e): raise
                r["measured_epoch_time_s"] = None
            free_cuda()
    ok = [r for r in rows if r["ok"]]
    out = {"gpu": gpu_name(), "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
           "row_range": rr, "cfg": asdict(cfg), "rows": rows,
           "max_batch": ok[-1]["batch_size"] if ok else None,
           "first_oom_batch": rows[-1]["batch_size"] if not rows[-1]["ok"] else None}
    save_json(out, f"{cfg.out}/{card_slug()}/partA.json")
    print(f"Max batch on {out['gpu']}: {out['max_batch']} (OOM at {out['first_oom_batch']})")


if __name__ == "__main__":
    main()
