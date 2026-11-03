"""HW4.a part 2 + HW4.5 Part D: vLLM vs Hugging Face pipeline serving benchmark, and KV-cache sizing.

Run each mode in its own process (vLLM grabs most of the GPU memory):
  python serve_bench.py --mode hf    --model Qwen/Qwen2.5-0.5B-Instruct
  python serve_bench.py --mode vllm  --model Qwen/Qwen2.5-0.5B-Instruct
  python serve_bench.py --mode kv    --model Qwen/Qwen2.5-0.5B-Instruct     # Part D KV cache

Method (same for every row of Table HW4.1):
  * identical prompts, greedy decoding, exactly --new-tokens generated per request
    (ignore_eos / min_new_tokens) so tokens/s is comparable;
  * a "request" is one prompt; offline batches are submitted together, so each request's latency
    is the wall time of the batch it was in (all requests finish on the same step because they all
    generate the same number of tokens);
  * --iters timed batches after --warmup untimed ones; p50/p95 over all per-request latencies;
  * tokens/s = total generated tokens / total timed wall time.
"""
import argparse, json, os, time
import numpy as np
import torch

PROMPTS = [
    "Explain in two sentences why the sky is blue.",
    "Write a short tweet announcing a new coffee shop opening downtown.",
    "Summarise the plot of a heist movie in three sentences.",
    "Give three tips for staying focused while studying.",
    "What is the difference between a GPU and a CPU?",
    "Describe a rainy afternoon in a small seaside town.",
    "List four uses of a paperclip other than holding paper.",
    "Why do neural networks need non-linear activation functions?",
]


def prompts_for(n, i):
    return [PROMPTS[(i * n + k) % len(PROMPTS)] + f" (#{i}-{k})" for k in range(n)]


def stats(lat_s, toks, wall):
    lat = np.array(lat_s) * 1000
    return {"tokens_per_s": toks / wall, "p50_ms": float(np.percentile(lat, 50)),
            "p95_ms": float(np.percentile(lat, 95)), "mean_ms": float(lat.mean()),
            "requests": len(lat), "gen_tokens": int(toks), "wall_s": wall}


def gpu():
    return torch.cuda.get_device_name(0).replace("NVIDIA ", "").replace("GeForce ", "").replace(" ", "_")


def save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=2); print("[saved]", path)


def chat(tok, p):
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
    return p


def run_hf(a):
    from transformers import pipeline, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    dtype = torch.bfloat16 if (a.dtype == "auto" and torch.cuda.is_bf16_supported()) else torch.float16
    pipe = pipeline("text-generation", model=a.model, tokenizer=tok, torch_dtype=dtype, device=0)
    kw = dict(max_new_tokens=a.new_tokens, min_new_tokens=a.new_tokens, do_sample=False, return_full_text=False)
    for i in range(a.warmup):
        pipe(chat(tok, prompts_for(1, i)[0]), **kw)
    lat, toks = [], 0
    torch.cuda.synchronize(); T = time.perf_counter()
    for i in range(a.iters):
        p = chat(tok, prompts_for(1, i + 100)[0])
        t = time.perf_counter(); out = pipe(p, **kw); torch.cuda.synchronize()
        lat.append(time.perf_counter() - t)
        toks += len(tok(out[0]["generated_text"], add_special_tokens=False).input_ids)
    wall = time.perf_counter() - T
    r = {"HF pipeline, batch 1": stats(lat, toks, wall)}
    r["_meta"] = {"model": a.model, "dtype": str(dtype), "gpu": gpu(), "new_tokens": a.new_tokens}
    print(json.dumps(r, indent=2)); save(r, f"{a.out}/{gpu()}/serve_hf.json")


def make_llm(a, **kw):
    from vllm import LLM
    return LLM(model=a.model, dtype=a.dtype, gpu_memory_utilization=a.gpu_util, seed=0,
               max_model_len=kw.pop("max_model_len", a.max_model_len), enforce_eager=a.eager, **kw)


def run_vllm(a):
    from vllm import SamplingParams
    llm = make_llm(a)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0, max_tokens=a.new_tokens, ignore_eos=True)
    res = {}
    for bs in a.batches:
        for i in range(a.warmup):
            llm.generate([chat(tok, p) for p in prompts_for(bs, i)], sp, use_tqdm=False)
        lat, toks = [], 0
        T = time.perf_counter()
        for i in range(a.iters):
            ps = [chat(tok, p) for p in prompts_for(bs, i + 100)]
            t = time.perf_counter(); outs = llm.generate(ps, sp, use_tqdm=False); dt = time.perf_counter() - t
            lat += [dt] * bs
            toks += sum(len(o.outputs[0].token_ids) for o in outs)
        res[f"vLLM, batch {bs}"] = stats(lat, toks, time.perf_counter() - T)
        print(bs, res[f"vLLM, batch {bs}"])
    res["_meta"] = {"model": a.model, "dtype": a.dtype, "gpu": gpu(), "new_tokens": a.new_tokens,
                    "gpu_memory_utilization": a.gpu_util, "kv_cache": kv_capacity(llm)}
    save(res, f"{a.out}/{gpu()}/serve_vllm.json")


def kv_capacity(llm):
    """Number of KV-cache tokens vLLM allocated (works on V0 and V1 engines when exposed)."""
    for path in ("llm_engine.vllm_config.cache_config", "llm_engine.cache_config"):
        o = llm
        try:
            for p in path.split("."):
                o = getattr(o, p)
            if getattr(o, "num_gpu_blocks", None):
                return {"num_gpu_blocks": o.num_gpu_blocks, "block_size": o.block_size,
                        "kv_tokens": o.num_gpu_blocks * o.block_size}
        except AttributeError:
            continue
    return {"note": "not exposed by this vLLM version -- read '# GPU blocks' / 'KV cache size' from the startup log"}


def run_kv(a):
    """Part D: KV-cache memory vs context length at fixed batch, and max context at batch 8.

    1. Analytic bytes/token from the model config: 2 (K,V) * layers * kv_heads * head_dim * dtype_bytes.
    2. Measured: HF forward pass with use_cache at batch=a.kv_batch for each context length;
       KV memory = size of past_key_values tensors (and torch peak-memory delta as a cross-check).
       Runs until OOM, which gives the HF (no paging) limit.
    3. vLLM: KV token capacity at --gpu-util; longest context at batch 8 = capacity / 8 (capped by
       the model's max positions). Then verified by actually generating 8 requests at that length.
    """
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(a.model)
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    kvh = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    dtype = torch.bfloat16 if (a.dtype == "auto" and torch.cuda.is_bf16_supported()) else torch.float16
    bpt = 2 * cfg.num_hidden_layers * kvh * hd * 2
    out = {"model": a.model, "gpu": gpu(), "batch": a.kv_batch, "bytes_per_token_analytic": bpt,
           "max_position_embeddings": cfg.max_position_embeddings,
           "total_vram_gb": torch.cuda.get_device_properties(0).total_memory / 1e9, "hf_measured": []}

    m = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=dtype).cuda().eval()
    L = 256
    while L <= cfg.max_position_embeddings:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        try:
            ids = torch.randint(100, 1000, (a.kv_batch, L), device="cuda")
            with torch.no_grad():
                o = m(input_ids=ids, use_cache=True, logits_to_keep=1) if _has_ltk(m) else m(input_ids=ids, use_cache=True)
            pkv = o.past_key_values
            kv_bytes = sum(t.numel() * t.element_size() for t in _iter_kv(pkv))
            rec = {"context": L, "kv_gb_measured": kv_bytes / 1e9,
                   "kv_gb_analytic": bpt * L * a.kv_batch / 1e9,
                   "peak_delta_gb": (torch.cuda.max_memory_allocated() - base) / 1e9, "ok": True}
            del o, pkv, ids
        except torch.cuda.OutOfMemoryError:
            rec = {"context": L, "ok": False, "error": "OOM"}
        print(rec); out["hf_measured"].append(rec)
        if not rec["ok"]:
            break
        L *= 2
    del m; torch.cuda.empty_cache()

    # vLLM capacity
    from vllm import SamplingParams
    llm = make_llm(a, max_model_len=min(cfg.max_position_embeddings, a.kv_max_model_len))
    cap = kv_capacity(llm); out["vllm_kv"] = cap
    if "kv_tokens" in cap:
        out["vllm_kv"]["kv_gb"] = cap["kv_tokens"] * bpt / 1e9
        mx = min(cap["kv_tokens"] // 8, llm.llm_engine.model_config.max_model_len)
        out["vllm_longest_context_batch8"] = mx
        # verify: 8 requests whose prompt+output fill mx tokens
        tok = llm.get_tokenizer()
        plen = mx - 16
        prompt_ids = [[(100 + j + k) % 5000 + 100 for j in range(plen)] for k in range(8)]
        t = time.perf_counter()
        try:
            from vllm.inputs import TokensPrompt
            prompts = [TokensPrompt(prompt_token_ids=p) for p in prompt_ids]
        except ImportError:
            prompts = [{"prompt_token_ids": p} for p in prompt_ids]
        outs = llm.generate(prompts, SamplingParams(max_tokens=16, ignore_eos=True), use_tqdm=False)
        out["vllm_verify_batch8"] = {"context": mx, "completed": len(outs), "wall_s": time.perf_counter() - t}
    save(out, f"{a.out}/{gpu()}/kv_cache.json")


def _has_ltk(m):
    import inspect
    return "logits_to_keep" in inspect.signature(m.forward).parameters


def _iter_kv(pkv):
    if hasattr(pkv, "layers"):                      # transformers >= 4.54 Cache objects
        for l in pkv.layers:
            yield l.keys; yield l.values
    elif hasattr(pkv, "key_cache"):
        yield from pkv.key_cache; yield from pkv.value_cache
    else:
        for k, v in pkv:
            yield k; yield v


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["hf", "vllm", "kv"], required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dtype", default="auto")   # use "half" on T4/P100 (no bf16)
    p.add_argument("--new-tokens", type=int, default=128)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32])
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--gpu-util", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--kv-max-model-len", type=int, default=32768)
    p.add_argument("--kv-batch", type=int, default=8)
    p.add_argument("--eager", action="store_true")
    p.add_argument("--out", default="results")
    a = p.parse_args()
    {"hf": run_hf, "vllm": run_vllm, "kv": run_kv}[a.mode](a)
