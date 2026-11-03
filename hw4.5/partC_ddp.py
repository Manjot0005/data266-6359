"""HW4.5 Part C: two-GPU DistributedDataParallel on the HW4 fine-tune.

  # single-GPU baseline through the same code path
  torchrun --nproc_per_node=1 partC_ddp.py --csv ... --slice $SLICE --seed $SEED --per-gpu-batches 8 16 32 64 128
  # two GPUs, one node (e.g. Kaggle 2xT4 = homogeneous, single-node, PCIe)
  torchrun --nproc_per_node=2 partC_ddp.py ... --per-gpu-batches 8 16 32 64 128
  # two nodes: torchrun --nnodes=2 --node_rank=R --master_addr=... --master_port=29500 --nproc_per_node=1 ...

For each per-GPU batch size:
  * throughput (global samples/s) over --steps timed steps after warm-up
  * per-step time WITH gradient all-reduce vs WITHOUT (model.no_sync()): the difference is the
    exposed communication overhead per step (what overlap with backward fails to hide)
  * per-rank compute time (to see a slow worker in a heterogeneous pair) and per-rank wait time
  * a standalone NCCL all-reduce of a gradient-sized buffer -> effective algbw / busbw (GB/s)
Writes results/<card>/partC_ws<world>.json (rank 0) and prints the interconnect topology.
"""
import os, time, json, subprocess
from dataclasses import asdict
import torch, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler
from common import Cfg, build_datasets, build_model, build_optim, autocast_ctx, save_json, card_slug, set_seed


def extra(p):
    p.add_argument("--per-gpu-batches", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warmup", type=int, default=5)


def allreduce_bw(nbytes, iters=20):
    x = torch.empty(nbytes // 4, dtype=torch.float32, device="cuda")
    for _ in range(3): dist.all_reduce(x)
    torch.cuda.synchronize(); dist.barrier(); t = time.perf_counter()
    for _ in range(iters): dist.all_reduce(x)
    torch.cuda.synchronize(); dt = (time.perf_counter() - t) / iters
    n = dist.get_world_size()
    algbw = nbytes / dt / 1e9
    return {"bytes": nbytes, "time_ms": dt * 1000, "algbw_GBps": algbw, "busbw_GBps": algbw * 2 * (n - 1) / n}


def timed(model, opt, dl_iter, cfg, steps, sync):
    """Return (wall_per_step_s, compute_per_step_s) for `steps` steps; sync=False uses no_sync()."""
    times, comp = [], []
    for _ in range(steps):
        ids, am, y = [t.cuda(non_blocking=True) for t in next(dl_iter)]
        torch.cuda.synchronize(); t0 = time.perf_counter()
        ctx = model.no_sync() if (not sync and isinstance(model, DDP)) else torch.enable_grad()
        with ctx:
            with autocast_ctx(cfg):
                loss = model(input_ids=ids, attention_mask=am, labels=y).loss
            loss.backward()
        torch.cuda.synchronize(); t1 = time.perf_counter()
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); t2 = time.perf_counter()
        times.append(t2 - t0); comp.append(t1 - t0)
    return sum(times) / steps, sum(comp) / steps


def main():
    cfg, a = Cfg.from_cli(extra)
    dist.init_process_group("nccl")
    rank, ws, local = dist.get_rank(), dist.get_world_size(), int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local)
    splits, _, names, rr, _ = build_datasets(cfg)
    train = splits["train"]

    model = build_model(cfg, len(names), "cuda")
    grad_bytes = sum(p.numel() * 4 for p in model.parameters() if p.requires_grad)
    if ws > 1:
        model = DDP(model, device_ids=[local])
    opt = build_optim(cfg, model)

    names_all = [None] * ws
    dist.all_gather_object(names_all, torch.cuda.get_device_name(local))
    info = {"world_size": ws, "gpus": names_all,
            "homogeneous": len(set(names_all)) == 1,
            "nodes": int(os.environ.get("GROUP_WORLD_SIZE", os.environ.get("NNODES", 1)) or 1),
            "grad_bytes_fp32": grad_bytes, "row_range": rr, "cfg": asdict(cfg)}
    if rank == 0:
        try:
            info["nvidia_smi_topo"] = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True).stdout
            print(info["nvidia_smi_topo"])
        except Exception as e:
            info["nvidia_smi_topo"] = str(e)
        info["nccl_env"] = {k: v for k, v in os.environ.items() if k.startswith("NCCL")}

    rows = []
    for bs in a.per_gpu_batches:
        g = torch.Generator().manual_seed(cfg.seed + rank)
        n = (a.warmup + 2 * a.steps + 2) * bs
        dl = DataLoader(train, batch_size=bs, sampler=RandomSampler(train, replacement=True, num_samples=n, generator=g),
                        drop_last=True, pin_memory=True)
        it = iter(dl)
        try:
            timed(model, opt, it, cfg, a.warmup, sync=True)
            t_sync, c_sync = timed(model, opt, it, cfg, a.steps, sync=True)
            t_nosync, c_nosync = timed(model, opt, it, cfg, a.steps, sync=False)
            ok = True
        except torch.cuda.OutOfMemoryError:
            ok = False
        flag = torch.tensor([int(ok)], device="cuda"); dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        if not flag.item():
            if rank == 0: rows.append({"per_gpu_batch": bs, "ok": False, "error": "OOM"})
            torch.cuda.empty_cache(); break
        per = torch.tensor([t_sync, c_sync, t_nosync, c_nosync], device="cuda")
        gathered = [torch.zeros_like(per) for _ in range(ws)]
        dist.all_gather(gathered, per)
        if rank == 0:
            g = [x.tolist() for x in gathered]
            step = max(x[0] for x in g)                     # global step time = slowest rank
            nosync = max(x[2] for x in g)
            comp = [x[3] for x in g]                         # pure compute per rank (no all-reduce)
            rec = {"per_gpu_batch": bs, "global_batch": bs * ws, "ok": True,
                   "step_time_ms": step * 1000, "step_time_nosync_ms": nosync * 1000,
                   "comm_overhead_ms": (step - nosync) * 1000,
                   "comm_fraction": (step - nosync) / step,
                   "compute_ms_per_rank": [c * 1000 for c in comp],
                   "slowest_minus_fastest_compute_ms": (max(comp) - min(comp)) * 1000,
                   "throughput_samples_per_s": bs * ws / step,
                   "grad_allreduce_effective_GBps": grad_bytes / max(step - nosync, 1e-9) / 1e9 if ws > 1 else None,
                   "dominant": ("communication" if (step - nosync) > nosync else "computation") if ws > 1 else "computation"}
            print(rec); rows.append(rec)
    bw = allreduce_bw(grad_bytes) if ws > 1 else None
    if rank == 0:
        info.update(rows=rows, allreduce_benchmark=bw)
        save_json(info, f"{cfg.out}/{card_slug()}/partC_ws{ws}.json")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
