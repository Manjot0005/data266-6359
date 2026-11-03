"""Build every table/plot for HW4.a + HW4.5 from results/<card>/*.json, and write METRICS.md.

  python report.py --results results --full-dataset-rows 1600000 --epochs 4
Outputs: REPORT_TABLES.md, METRICS.md, plots/*.png. Nothing is invented: a missing JSON -> "n/a".
"""
import argparse, glob, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(results):
    d = {}
    for f in glob.glob(f"{results}/*/*.json"):
        card = os.path.basename(os.path.dirname(f))
        d.setdefault(card, {})[os.path.basename(f)[:-5]] = json.load(open(f))
    return d


def f(x, n=2):
    return "n/a" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def md_table(head, rows):
    s = "| " + " | ".join(head) + " |\n|" + "---|" * len(head) + "\n"
    return s + "".join("| " + " | ".join(map(str, r)) + " |\n" for r in rows)


def hw4a(card, D, out):
    t = D.get("hw4a_train")
    if not t: return
    rr = t["row_range"]
    out.append(f"## HW4.a — {card}\n\nRows used: **{rr['first_row']}–{rr['last_row']}** ({rr['n_rows']} rows of "
               f"{rr['dataset_rows']}); split {rr['split_sizes']}; labels {rr['label_counts']}\n")
    out.append(md_table(["Epoch", "Train loss", "Val loss", "Val acc", "Epoch time (s)"],
               [[h["epoch"], f(h["train_loss"], 4), f(h.get("val_loss"), 4), f(h.get("val_acc"), 4), f(h["epoch_time_s"], 1)]
                for h in t["history"]]))
    out.append(f"\nValidation loss stopped improving after epoch **{t['best_val_epoch']}**; "
               f"{t['epochs_past_best']} extra epoch(s) cost {f(t['extra_time_past_best_s'],1)} s of training "
               f"(compare val loss/acc of the last epoch vs the best one above for the accuracy cost).\n")
    cr = t["class_report"]; L = cr["labels"]
    out.append("\nConfusion matrix (rows = true, cols = predicted):\n\n" +
               md_table(["true \\ pred"] + L, [[L[i]] + r for i, r in enumerate(cr["confusion_matrix"])]))
    out.append("\n" + md_table(["Class", "Precision", "Recall", "F1", "Support"],
               [[k, f(v["precision"], 3), f(v["recall"], 3), f(v["f1"], 3), v["support"]] for k, v in cr["per_class"].items()]))
    out.append(f"\nWorst class: **{cr['worst_class']}**. Misclassified examples:\n\n" +
               md_table(["Text", "Gold", "Pred", "Diagnosis"],
                        [[m["text"].replace("|", "/")[:200], m["gold"], m["pred"], m["diagnosis"]] for m in t["worst_class_misclassified"]]))
    out.append(f"\n**HW4 baseline (for HW4.5 only):** peak GPU memory {f(t['peak_mem_alloc_gb'])} GB allocated "
               f"({f(t['peak_mem_reserved_gb'])} GB reserved), training wall-clock {f(t['train_time_s'],1)} s "
               f"({f(t['wall_clock_incl_eval_s'],1)} s incl. validation), test acc {f(t['test_acc'],4)}.\n")


def serving(card, D, out, label="HW4.1"):
    hf, vl = D.get("serve_hf", {}), D.get("serve_vllm", {})
    if not (hf or vl): return None
    rows, all_ = [], {**{k: v for k, v in hf.items() if not k.startswith("_")}, **{k: v for k, v in vl.items() if not k.startswith("_")}}
    for k in ["HF pipeline, batch 1", "vLLM, batch 1", "vLLM, batch 8", "vLLM, batch 32"]:
        r = all_.get(k, {})
        rows.append([k, f(r.get("tokens_per_s"), 1), f(r.get("p50_ms"), 1), f(r.get("p95_ms"), 1)])
    meta = vl.get("_meta") or hf.get("_meta") or {}
    out.append(f"\n### Table {label} — Serving comparison ({card}, model {meta.get('model')}, {meta.get('new_tokens')} new tokens)\n\n"
               + md_table(["Configuration", "Tokens/sec", "p50 latency (ms)", "p95 latency (ms)"], rows))
    base = all_.get("HF pipeline, batch 1")
    if base:
        prev = None
        for bs in (1, 8, 32):
            r = all_.get(f"vLLM, batch {bs}")
            if r and r["p50_ms"] > base["p50_ms"] and (prev is None or r["tokens_per_s"] > prev):
                out.append(f"\nCrossover: at **vLLM batch {bs}** p50 per-request latency ({f(r['p50_ms'],0)} ms) exceeds the HF "
                           f"batch-1 baseline ({f(base['p50_ms'],0)} ms) while total throughput is still rising "
                           f"({f(r['tokens_per_s'],0)} tok/s).\n")
                break
            prev = r["tokens_per_s"] if r else prev
        else:
            out.append("\nNo crossover in 1/8/32: vLLM per-request latency stayed below the HF batch-1 baseline.\n")
    return all_


def partA(D_all, out, plots):
    cards = {c: D["partA"] for c, D in D_all.items() if "partA" in D}
    if not cards: return
    out.append("\n## Part A — Batch scaling to the memory wall\n")
    plt.figure(figsize=(6, 4))
    for c, A in cards.items():
        out.append(f"\n**{c}** ({f(A['vram_gb'],1)} GB): max batch **{A['max_batch']}**, OOM at {A['first_oom_batch']}\n\n" +
                   md_table(["Batch", "Samples/s", "Peak alloc (GB)", "Peak reserved (GB)", "Epoch time (s)"],
                            [[r["batch_size"], f(r.get("samples_per_s"), 1), f(r.get("peak_mem_alloc_gb")),
                              f(r.get("peak_mem_reserved_gb")),
                              f(r.get("measured_epoch_time_s") or r.get("est_epoch_time_s"), 1)] if r["ok"]
                             else [r["batch_size"], "OOM", "", "", ""] for r in A["rows"]]))
        ok = [r for r in A["rows"] if r["ok"]]
        xs, ys = [r["batch_size"] for r in ok], [r["samples_per_s"] for r in ok]
        plt.plot(xs, ys, "o-", label=c)
        # knee = smallest batch whose next doubling gains < 5% throughput
        knee = next((xs[i - 1] for i in range(1, len(ys)) if ys[i] < ys[i - 1] * 1.05), xs[-1])
        out.append(f"\nThroughput plateau (next doubling gains <5%): batch **{knee}** on {c}.\n")
    plt.xscale("log", base=2); plt.xlabel("Batch size"); plt.ylabel("Samples / s"); plt.title("BERT fine-tune throughput vs batch")
    plt.grid(alpha=.3); plt.legend(); os.makedirs("plots", exist_ok=True); plt.tight_layout(); plt.savefig("plots/partA_throughput.png", dpi=150)
    plots.append("plots/partA_throughput.png"); out.append("\n![](plots/partA_throughput.png)\n")
    if len(cards) >= 2:
        (c1, A1), (c2, A2) = sorted(cards.items(), key=lambda kv: -kv[1]["vram_gb"])[:2]
        out.append(f"\nMax-batch ratio {c1}/{c2} = {A1['max_batch']}/{A2['max_batch']} = **{A1['max_batch']/A2['max_batch']:.2f}**; "
                   f"VRAM ratio = {f(A1['vram_gb'],1)}/{f(A2['vram_gb'],1)} = **{A1['vram_gb']/A2['vram_gb']:.2f}**. "
                   "(Powers-of-two probing quantises the batch ratio; fixed costs — weights, grads, Adam state, CUDA context, "
                   "allocator fragmentation — are paid before any activation memory, so only the remainder scales with batch.)\n")


def partB(card, D, out):
    B = D.get("partB")
    if not B: return None
    R = B["results"]; base = R.get("baseline")
    out.append(f"\n## Part B — Memory techniques ({card})\n\n")
    order = ["baseline", "amp", "ckpt", "accum", "optim8", "combo"]
    rows = []
    for k in order:
        r = R.get(k)
        if not r: continue
        mb = r.get("max_batch")
        if k == "accum": mb = f"{r.get('max_batch')} phys → eff {r.get('effective_batch')}"
        rows.append([r["technique"], mb, f(r["samples_per_s"], 1), f(r["peak_mem_alloc_gb"]), f(r["test_acc"], 4)])
    out.append(md_table(["Technique", "Max batch", "Throughput (samp/s)", "Peak memory (GB)", "Eval accuracy"], rows))
    out.append(f"\nThroughput and peak memory are measured at the HW4 reference batch ({base['ref_batch'] if base else 'n/a'}) so they are comparable.\n")
    if base and "ckpt" in R:
        c = R["ckpt"]; out.append(f"\nGradient checkpointing throughput cost: {100*(1-c['samples_per_s']/base['samples_per_s']):.1f}% "
                                  f"({f(base['samples_per_s'],1)} → {f(c['samples_per_s'],1)} samp/s).\n")
    if "accum" in R:
        a = R["accum"]; p = a.get("physical_large_batch", {})
        out.append(f"\nGradient accumulation: effective batch {a.get('effective_batch')} from physical {a['ref_batch']}; "
                   f"{a['opt_steps']} optimiser steps in {f(a['train_time_s'],1)} s (acc {f(a['test_acc'],4)}) vs physical batch "
                   f"{p.get('batch')}: {f(p.get('train_time_s'),1)} s (acc {f(p.get('test_acc'),4)}) {p.get('error','')}.\n")
    if base:
        rk = []
        for k in ("amp", "ckpt", "accum", "optim8"):
            r = R.get(k)
            if not r: continue
            saved = base["peak_mem_alloc_gb"] - r["peak_mem_alloc_gb"]
            lost = 100 * (1 - r["samples_per_s"] / base["samples_per_s"])
            score = saved / max(lost, 0.5) if saved > 0 else saved   # floor 0.5% avoids div-by-~0; speed-ups rank top
            rk.append((score, r["technique"], saved, lost))
        rk.sort(reverse=True)
        out.append("\n**Ranking — memory saved per unit throughput lost** (GB saved ÷ % throughput lost, loss floored at 0.5%):\n\n" +
                   md_table(["Rank", "Technique", "Memory saved (GB)", "Throughput lost (%)", "GB per % lost"],
                            [[i + 1, t, f(s), f(l, 1), f(sc, 3)] for i, (sc, t, s, l) in enumerate(rk)]))
    return R


def partC(D_all, out):
    for c, D in D_all.items():
        one, two = D.get("partC_ws1"), D.get("partC_ws2")
        if not two: continue
        out.append(f"\n## Part C — DDP ({', '.join(two['gpus'])})\n\n"
                   f"Pairing: {'homogeneous' if two['homogeneous'] else 'heterogeneous'}, "
                   f"{'single-node' if two['nodes'] == 1 else 'multi-node'}. Gradient size {two['grad_bytes_fp32']/1e6:.0f} MB (fp32).\n")
        b1 = {r["per_gpu_batch"]: r for r in (one or {}).get("rows", []) if r["ok"]}
        rows, worth = [], None
        for r in two["rows"]:
            if not r["ok"]: rows.append([r["per_gpu_batch"], "OOM"] + [""] * 6); continue
            s = b1.get(r["per_gpu_batch"])
            eff = r["throughput_samples_per_s"] / (2 * s["throughput_samples_per_s"]) if s else None
            spd = r["throughput_samples_per_s"] / s["throughput_samples_per_s"] if s else None
            if worth is None and spd and spd > 1.2: worth = r["per_gpu_batch"]
            rows.append([r["per_gpu_batch"], f(s["throughput_samples_per_s"], 1) if s else "n/a", f(r["throughput_samples_per_s"], 1),
                         f(eff, 2), f(r["comm_overhead_ms"], 1), f(100 * r["comm_fraction"], 1),
                         f(r["grad_allreduce_effective_GBps"], 2), r["dominant"]])
        out.append(md_table(["Per-GPU batch", "1-GPU samp/s", "2-GPU samp/s", "Scaling eff.", "Comm/step (ms)", "Comm %",
                             "Eff. grad BW (GB/s)", "Dominates"], rows))
        bw = two.get("allreduce_benchmark") or {}
        out.append(f"\nStandalone NCCL all-reduce of the gradient buffer: {f(bw.get('time_ms'),1)} ms, algbw {f(bw.get('algbw_GBps'))} GB/s, "
                   f"busbw {f(bw.get('busbw_GBps'))} GB/s.\n\nFirst per-GPU batch with >1.2× speed-up over 1 GPU: **{worth}**.\n"
                   f"\nTopology:\n```\n{two.get('nvidia_smi_topo','')}\n```\n")


def kv(card, D, out):
    k = D.get("kv_cache")
    if not k: return
    out.append(f"\n## Part D — KV cache ({card}, {k['model']}, batch {k['batch']})\n\n"
               f"Analytic KV bytes/token: {k['bytes_per_token_analytic']} ({k['bytes_per_token_analytic']/1024:.0f} KiB)\n\n" +
               md_table(["Context", "KV measured (GB)", "KV analytic (GB)", "Forward peak Δ (GB)"],
                        [[r["context"], f(r.get("kv_gb_measured"), 3), f(r.get("kv_gb_analytic"), 3), f(r.get("peak_delta_gb"), 2)] if r["ok"]
                         else [r["context"], "OOM", "", ""] for r in k["hf_measured"]]))
    v = k.get("vllm_kv", {})
    out.append(f"\nvLLM KV pool: {v.get('kv_tokens')} tokens ({f(v.get('kv_gb'))} GB). Longest context at batch 8: "
               f"**{k.get('vllm_longest_context_batch8')}** tokens; verification run: {k.get('vllm_verify_batch8')}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results")
    p.add_argument("--full-dataset-rows", type=int, default=1_600_000)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--train-frac", type=float, default=0.70)
    a = p.parse_args()
    D_all = load(a.results); out, plots = ["# HW4.a / HW4.5 results (generated by report.py)\n"], []
    for c, D in D_all.items(): hw4a(c, D, out)
    for c, D in D_all.items(): serving(c, D, out, "HW4.1 / Part D")
    partA(D_all, out, plots)
    best = {}
    for c, D in D_all.items():
        R = partB(c, D, out)
        if R: best[c] = R
    partC(D_all, out)
    for c, D in D_all.items(): kv(c, D, out)

    # Part E: GPU-hours
    out.append("\n## Part E — GPU-hours for one full fine-tune\n")
    m = ["# METRICS.md — HW4.5 summary\n", "_HW4.5 summary only; not an HW5 baseline._\n"]
    for c, R in best.items():
        cand = {k: v for k, v in R.items() if v}
        fastest = max(cand.values(), key=lambda r: r["samples_per_s"])
        leanest = min(cand.values(), key=lambda r: r["peak_mem_alloc_gb"])
        quickest = min(cand.values(), key=lambda r: r["train_time_s"])
        n = a.full_dataset_rows * a.train_frac * a.epochs
        for r in (R.get("baseline"), fastest):
            if not r: continue
            thr = r.get("full_run_samples_per_s") or r["samples_per_s"]
            out.append(f"\n{c}, {r['technique']}: {a.full_dataset_rows:,} rows × {a.train_frac} train × {a.epochs} epochs = "
                       f"{n:,.0f} samples ÷ {thr:.1f} samp/s = {n/thr:,.0f} s = **{n/thr/3600:.2f} GPU-hours** (training only, excl. eval).\n")
        m.append(f"\n## {c}\n\n" + md_table(["Metric", "Configuration", "Value"], [
            ["Best peak memory", leanest["technique"], f"{leanest['peak_mem_alloc_gb']:.2f} GB @ batch {leanest['ref_batch']}"],
            ["Best throughput", fastest["technique"], f"{fastest['samples_per_s']:.1f} samples/s @ batch {fastest['ref_batch']}"],
            ["Best wall-clock (full fine-tune)", quickest["technique"], f"{quickest['train_time_s']:.1f} s (test acc {quickest['test_acc']:.4f})"],
        ]))
        m.append(f"\nFull fine-tune estimate ({fastest['technique']}): {n/(fastest.get('full_run_samples_per_s') or fastest['samples_per_s'])/3600:.2f} GPU-hours.\n")
    open("REPORT_TABLES.md", "w").write("".join(out)); open("METRICS.md", "w").write("".join(m))
    print("wrote REPORT_TABLES.md, METRICS.md", plots)


if __name__ == "__main__":
    main()
