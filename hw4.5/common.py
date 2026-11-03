"""Shared code for HW4.a (BERT fine-tune) and HW4.5 (memory / throughput engineering).

Every experiment goes through `train_and_eval()` so the "HW4 settings" are identical
everywhere; techniques are switched on by flags, one at a time.
"""
import argparse, json, math, os, random, time, contextlib
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# ----------------------------------------------------------------------------- config
@dataclass
class Cfg:
    # data
    csv: str = ""                 # path to Twitter CSV; empty -> HF tweet_eval/sentiment
    text_col: str = "text"
    label_col: str = "label"
    csv_header: bool = True       # Sentiment140 has no header: use --no-header and col indices
    encoding: str = "latin-1"
    slice: int = 0                # SLICE (assigned)
    n_rows: int = 5000
    seed: int = 42                # SEED (assigned)
    max_len: int = 128
    # model / train (HW4 settings)
    model: str = "bert-base-uncased"
    epochs: int = 4
    batch_size: int = 32
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_frac: float = 0.1
    # techniques (HW4.5 Part B) -- all off == baseline
    amp: str = "none"             # none | fp16 | bf16
    grad_ckpt: bool = False
    accum: int = 1                # gradient accumulation steps
    optim: str = "adamw"          # adamw | adamw8bit | adafactor
    max_opt_steps: int = 0        # >0: stop after this many optimiser steps (Part B.3)
    # bookkeeping
    tag: str = "baseline"
    out: str = "results"
    save_model: str = ""

    @staticmethod
    def from_cli(extra=None):
        p = argparse.ArgumentParser()
        for k, v in asdict(Cfg()).items():
            if isinstance(v, bool):
                p.add_argument(f"--{k.replace('_','-')}", dest=k, action=argparse.BooleanOptionalAction, default=v)
            else:
                p.add_argument(f"--{k.replace('_','-')}", dest=k, type=type(v), default=v)
        if extra:
            extra(p)
        a = p.parse_args()
        cfg = Cfg(**{k: getattr(a, k) for k in asdict(Cfg())})
        return cfg, a


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def gpu_name():
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"


def card_slug():
    return gpu_name().replace("NVIDIA ", "").replace("GeForce ", "").replace(" ", "_")


def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"[saved] {path}")


# ----------------------------------------------------------------------------- data
def load_subset(cfg: Cfg):
    """Return (texts, labels, label_names, row_range) for up to n_rows consecutive rows from SLICE."""
    if cfg.csv:
        df = pd.read_csv(cfg.csv, encoding=cfg.encoding, header=0 if cfg.csv_header else None)
        tcol = cfg.text_col if cfg.csv_header else int(cfg.text_col)
        lcol = cfg.label_col if cfg.csv_header else int(cfg.label_col)
    else:
        from datasets import load_dataset
        df = load_dataset("cardiffnlp/tweet_eval", "sentiment", split="train").to_pandas()
        tcol, lcol = "text", "label"
    n_total = len(df)
    start = cfg.slice
    if start >= n_total:
        raise ValueError(f"SLICE={start} is past the end of the dataset ({n_total} rows)")
    end = min(start + cfg.n_rows, n_total)            # exclusive; uses all remaining if < n_rows
    sub = df.iloc[start:end].dropna(subset=[tcol, lcol])
    names = sorted(sub[lcol].unique().tolist())
    if len(names) < 2:
        raise ValueError(f"Rows {start}..{end-1} contain a single label ({names}). "
                         "Sentiment140 is sorted by label -- pick a SLICE that straddles classes "
                         "or confirm with the instructor.")
    lab2id = {n: i for i, n in enumerate(names)}
    texts = sub[tcol].astype(str).tolist()
    labels = sub[lcol].map(lab2id).astype(int).tolist()
    rr = {"first_row": start, "last_row": end - 1, "n_rows": end - start,
          "n_after_dropna": len(sub), "dataset_rows": n_total,
          "label_counts": {str(n): int((sub[lcol] == n).sum()) for n in names}}
    return texts, labels, [str(n) for n in names], rr


def split_70_15_15(n, seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    a, b = int(round(0.70 * n)), int(round(0.85 * n))
    return idx[:a], idx[a:b], idx[b:]


def build_datasets(cfg: Cfg):
    from transformers import BertTokenizer
    texts, labels, names, rr = load_subset(cfg)
    tok = BertTokenizer.from_pretrained(cfg.model)
    enc = tok(texts, padding="max_length", truncation=True, max_length=cfg.max_len, return_tensors="pt")
    y = torch.tensor(labels)
    tr, va, te = split_70_15_15(len(texts), cfg.seed)
    mk = lambda ix: TensorDataset(enc["input_ids"][ix], enc["attention_mask"][ix], y[ix])
    splits = {"train": mk(tr), "val": mk(va), "test": mk(te)}
    rr["split_sizes"] = {k: len(v) for k, v in splits.items()}
    return splits, tok, names, rr, [texts[i] for i in te]


# ----------------------------------------------------------------------------- model / optim
def build_model(cfg: Cfg, n_labels, device):
    from transformers import BertForSequenceClassification
    set_seed(cfg.seed)                               # SEED for training initialisation
    m = BertForSequenceClassification.from_pretrained(cfg.model, num_labels=n_labels).to(device)
    if cfg.grad_ckpt:
        m.gradient_checkpointing_enable()
        m.config.use_cache = False
    return m


def build_optim(cfg: Cfg, model):
    decay = [p for n, p in model.named_parameters() if not any(k in n for k in ("bias", "LayerNorm"))]
    nodecay = [p for n, p in model.named_parameters() if any(k in n for k in ("bias", "LayerNorm"))]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay},
              {"params": nodecay, "weight_decay": 0.0}]
    if cfg.optim == "adamw":
        return torch.optim.AdamW(groups, lr=cfg.lr)
    if cfg.optim == "adamw8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(groups, lr=cfg.lr)
    if cfg.optim == "adafactor":
        from transformers.optimization import Adafactor
        return Adafactor(groups, lr=cfg.lr, scale_parameter=False, relative_step=False)
    raise ValueError(cfg.optim)


def autocast_ctx(cfg: Cfg):
    if cfg.amp == "none":
        return contextlib.nullcontext()
    dt = torch.float16 if cfg.amp == "fp16" else torch.bfloat16
    return torch.autocast("cuda", dtype=dt)


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, ds, cfg, device, bs=128):
    model.eval()
    dl = DataLoader(ds, batch_size=bs, sampler=SequentialSampler(ds))
    tot, n, preds, gold = 0.0, 0, [], []
    for ids, am, y in dl:
        ids, am, y = ids.to(device), am.to(device), y.to(device)
        with autocast_ctx(cfg):
            out = model(input_ids=ids, attention_mask=am, labels=y)
        tot += out.loss.float().item() * len(y); n += len(y)
        preds += out.logits.argmax(-1).tolist(); gold += y.tolist()
    model.train()
    preds, gold = np.array(preds), np.array(gold)
    return {"loss": tot / n, "acc": float((preds == gold).mean()), "preds": preds.tolist(), "gold": gold.tolist()}


def class_report(gold, preds, names):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    cm = confusion_matrix(gold, preds, labels=list(range(len(names))))
    p, r, f, s = precision_recall_fscore_support(gold, preds, labels=list(range(len(names))), zero_division=0)
    per = {names[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
           for i in range(len(names))}
    worst = min(per, key=lambda k: per[k]["f1"])
    return {"confusion_matrix": cm.tolist(), "labels": names, "per_class": per, "worst_class": worst}


# ----------------------------------------------------------------------------- train
def train_and_eval(cfg: Cfg, splits, names, device="cuda", do_eval=True, timed_steps=None, quiet=False):
    """Fine-tune with cfg. Returns a metrics dict.

    timed_steps: if set, run only this many *micro-batches* (after 3 warm-up steps) on randomly
    sampled training rows and return throughput/peak-memory -- used by the OOM probe.
    """
    set_seed(cfg.seed)
    model = build_model(cfg, len(names), device)
    opt = build_optim(cfg, model)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp == "fp16"))
    train = splits["train"]

    if timed_steps:
        sampler = RandomSampler(train, replacement=True, num_samples=(timed_steps + 3) * cfg.batch_size)
        dl = DataLoader(train, batch_size=cfg.batch_size, sampler=sampler, drop_last=True)
        epochs = 1
    else:
        g = torch.Generator().manual_seed(cfg.seed)
        dl = DataLoader(train, batch_size=cfg.batch_size, sampler=RandomSampler(train, generator=g))
        epochs = cfg.epochs

    steps_per_epoch = math.ceil(len(dl) / cfg.accum)
    total_opt = cfg.max_opt_steps or steps_per_epoch * epochs
    from transformers import get_linear_schedule_with_warmup
    sched = get_linear_schedule_with_warmup(opt, int(cfg.warmup_frac * total_opt), total_opt)

    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    hist, opt_steps, samples, train_time = [], 0, 0, 0.0
    model.train()
    done = False
    for ep in range(epochs):
        t_ep = time.perf_counter(); run_loss, run_n, ep_samples = 0.0, 0, 0
        for i, (ids, am, y) in enumerate(dl):
            if timed_steps and i == 3:                   # end of warm-up: start the clock
                torch.cuda.synchronize(); t_ep = time.perf_counter(); ep_samples = 0
            ids, am, y = ids.to(device, non_blocking=True), am.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with autocast_ctx(cfg):
                loss = model(input_ids=ids, attention_mask=am, labels=y).loss / cfg.accum
            scaler.scale(loss).backward()
            run_loss += loss.item() * cfg.accum * len(y); run_n += len(y); ep_samples += len(y)
            if (i + 1) % cfg.accum == 0 or (i + 1) == len(dl):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
                opt_steps += 1
                if cfg.max_opt_steps and opt_steps >= cfg.max_opt_steps:
                    done = True; break
        torch.cuda.synchronize()
        ep_time = time.perf_counter() - t_ep
        train_time += ep_time; samples += ep_samples
        rec = {"epoch": ep + 1, "train_loss": run_loss / max(run_n, 1), "epoch_time_s": ep_time,
               "samples_per_s": ep_samples / ep_time}
        if do_eval and not timed_steps:
            v = evaluate(model, splits["val"], cfg, device)
            rec.update(val_loss=v["loss"], val_acc=v["acc"])
        hist.append(rec)
        if not quiet:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()})
        if done:
            break

    res = {"cfg": asdict(cfg), "gpu": gpu_name(), "history": hist,
           "peak_mem_alloc_gb": torch.cuda.max_memory_allocated() / 1e9,
           "peak_mem_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
           "train_time_s": train_time, "samples_per_s": samples / train_time,
           "opt_steps": opt_steps, "effective_batch": cfg.batch_size * cfg.accum}
    if timed_steps:
        res["est_epoch_time_s"] = len(train) / res["samples_per_s"]
    if do_eval and not timed_steps:
        vl = [h["val_loss"] for h in hist]
        best = int(np.argmin(vl))
        res["best_val_epoch"] = best + 1
        res["val_stopped_improving_after_epoch"] = best + 1
        res["epochs_past_best"] = len(hist) - (best + 1)
        res["extra_time_past_best_s"] = sum(h["epoch_time_s"] for h in hist[best + 1:])
        t = evaluate(model, splits["test"], cfg, device)
        res["test_acc"] = t["acc"]; res["test_loss"] = t["loss"]
        res["_test_preds"], res["_test_gold"] = t["preds"], t["gold"]
    res["_model"] = model
    return res


def is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def free_cuda():
    import gc
    gc.collect(); torch.cuda.empty_cache()


def probe_max_batch(cfg: Cfg, splits, names, start=8, limit=8192, steps=10, device="cuda"):
    """Double batch size until OOM. Returns list of per-batch records (last one = OOM)."""
    rows, bs = [], start
    while bs <= limit:
        c = Cfg(**{**asdict(cfg), "batch_size": bs})
        try:
            r = train_and_eval(c, splits, names, device, do_eval=False, timed_steps=steps, quiet=True)
            rec = {"batch_size": bs, "ok": True, "samples_per_s": r["samples_per_s"],
                   "peak_mem_alloc_gb": r["peak_mem_alloc_gb"], "peak_mem_reserved_gb": r["peak_mem_reserved_gb"],
                   "est_epoch_time_s": r["est_epoch_time_s"]}
            del r
        except Exception as e:
            if not is_oom(e):
                raise
            rec = {"batch_size": bs, "ok": False, "error": "OOM"}
        free_cuda()
        print(rec); rows.append(rec)
        if not rec["ok"]:
            break
        bs *= 2
    return rows
