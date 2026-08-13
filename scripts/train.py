"""Phase 3 -- pretrain the sequence-only baseline on the chr9 pilot.

Masked-language-model over nucleotides with BiMambaLM(structural=False). This is
the arm every later comparison is measured against, so it is trained and logged
*before* the structural model exists (CLAUDE.md Phase 3).

FIVE DECISIONS WORTH KNOWING BEFORE READING THE CODE
----------------------------------------------------

1. **The baseline runs the custom Triton scan even though it never supplies p.**
   `use_scan("triton")` is forced in every worker. The structural arm has no
   choice but to use this kernel (mamba_ssm cannot express the permeability
   term), so if the baseline used a different scan the matched-compute claim
   would be fiction. Cost is carried identically by both arms.

2. **fp32 throughout, no autocast.** scripts/validate_kernel.py validated the
   scan in fp32 only. Running it in bf16 would put the experiment on an
   unvalidated numeric path through a hand-written recurrence, which is exactly
   the failure the kernel validation existed to prevent.

3. **N is excluded from masking and from the loss.** 12.00% of chr9 is N and a
   window may carry up to 10% (data_card.md 4D.1). Predicting N is free, so
   including it would deflate bits/nucleotide by an amount that depends on how
   much N a window happens to contain. Only ACGT (ids 2-5) are maskable and only
   masked ACGT positions contribute loss, so the reported number is bits per
   *resolved* nucleotide and the uniform-random floor is exactly 2.000 bits.

4. **tau is measured two ways, and they are not interchangeable.**
   `BiMambaLM.tau_stats()` derives tau from `dt_proj.bias` alone. That is a good
   proxy at initialisation, where `dt_proj.weight` is small, but after training
   Delta is dominated by the input-dependent term `W_dt delta'_t` and a
   bias-only tau can be badly wrong. So this script logs `tau_stats()` because
   the runbook asks for it, and separately measures *empirical* tau from the
   Delta actually produced on validation batches. The Phase 4 gate in
   architecture_spec.md 4.1.4 F4 should be read off the empirical numbers.

5. **num_workers defaults to 0.** WindowDataset holds its own
   `np.random.default_rng` for reverse-complement augmentation. Under multiple
   loader workers each worker copies that generator, so the augmentation stream
   depends on how the sampler happened to shard -- reproducible seeds would stop
   being reproducible. Loading is a memmap slice and is not the bottleneck.

Run a smoke test first:
    python scripts/train.py --smoke

Then the real runs:
    python scripts/train.py --seed 0
    python scripts/train.py --seed 1
    python scripts/train.py --seed 2

Nothing in results/ may be quoted anywhere unless it came out of a finished run.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from chromfm.model import BiMambaLM, ModelConfig, use_scan   # noqa: E402
from phase1_dataset import WindowDataset                     # noqa: E402

RESULTS = REPO / "results" / "baselines"

# vocabulary, from phase1_dataset.py -- kept as literals so a silent change
# there shows up as a failing assertion rather than as a quietly different run
PAD, MASK_ID = 0, 1
NUC_LO, NUC_HI = 2, 5          # A, C, G, T inclusive
N_TOK = 6
LN2 = math.log(2.0)


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

class CheckpointedLM(torch.nn.Module):
    """BiMambaLM with per-layer gradient checkpointing.

    MEASURED NOT TO HELP HERE, and kept only because it is the lever to reach for
    if d_model or the window grows. The hypothesis was that the scan is
    latency-bound -- at batch 2 the grid is batch*d_inner = 1024 programs, about
    7 warps per SM on an L40S that can host far more -- so a bigger batch bought
    with recompute should have been close to free. It is not: batch scaling is
    linear, so the GPU is already saturated at batch 2. Measured on one L40S,
    1.32 s/window at batch 2 without checkpointing against 1.49 s/window at
    batch 8 with it, i.e. 13% worse, which is the recompute cost showing up
    undiluted.

    Wraps rather than mutates: `core` stays an unmodified BiMambaLM, so
    `tau_stats()`, the DeltaCapture hooks and the checkpoint state_dict all keep
    working on the real module tree.
    """

    def __init__(self, core: BiMambaLM):
        super().__init__()
        self.core = core

    def forward(self, tokens, phi=None, phi_valid=None, symmetry=None):
        c = self.core
        u = c.embed(tokens)
        s = s_rev = None
        if c.c.structural:                      # Phase 4 will need this path
            raise NotImplementedError(
                "checkpointed forward is wired for the sequence-only baseline; "
                "the structural arm must extend it to carry s and s_rev")
        for layer in c.layers:
            u = torch.utils.checkpoint.checkpoint(
                layer, u, s, s_rev, use_reentrant=False)
        return c.lm_head(c.norm_f(u))


def unwrap(model) -> BiMambaLM:
    """Peel DDP and CheckpointedLM down to the BiMambaLM itself."""
    m = model
    while not isinstance(m, BiMambaLM):
        if isinstance(m, DDP):
            m = m.module
        elif isinstance(m, CheckpointedLM):
            m = m.core
        else:
            raise TypeError(f"cannot unwrap {type(m)}")
    return m


def plain(obj):
    """Coerce to types yaml.safe_dump can represent. torch.__version__ is a str
    *subclass*, numpy scalars are not Python scalars, and SafeDumper refuses
    both -- which would otherwise blow up after the run rather than before it."""
    if isinstance(obj, dict):
        return {str(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, str):
        return str(obj)
    return str(obj)


def nvml_works() -> bool:
    """True if NVML initialises. NCCL needs it for topology discovery, so when
    this is False NCCL will fail at its first collective regardless of CUDA
    being healthy."""
    import ctypes
    try:
        return ctypes.CDLL("libnvidia-ml.so.1").nvmlInit_v2() == 0
    except Exception:
        return False


def git_state() -> dict:
    def run(*a):
        try:
            return subprocess.run(a, cwd=REPO, capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except Exception:
            return "unavailable"
    dirty = run("git", "status", "--porcelain")
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(dirty),
        "dirty_files": [ln[3:] for ln in dirty.splitlines()] if dirty else [],
    }


def env_state() -> dict:
    import triton
    gpus = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        gpus.append({
            "index": i, "name": p.name,
            "capability": f"{p.major}.{p.minor}",
            "total_memory_GiB": round(p.total_memory / 2**30, 2),
        })
    # nvidia-smi is unusable on this box (NVML/kernel-module version mismatch),
    # so the driver version is read from the loaded kernel module instead.
    try:
        driver = Path("/sys/module/nvidia/version").read_text().strip()
    except Exception:
        driver = "unavailable"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "triton": triton.__version__,
        "numpy": np.__version__,
        "nvidia_kernel_module": driver,
        "nvidia_smi": "UNAVAILABLE -- NVML/kernel-module version mismatch",
        "gpus": gpus,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
    }


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def collate(batch: list[dict]) -> dict:
    return {
        "tokens": torch.from_numpy(
            np.stack([b["tokens"] for b in batch]).astype(np.int64)),
        "start": torch.tensor([b["start"] for b in batch], dtype=torch.int64),
    }


def make_loader(split: str, batch_size: int, rank: int, world: int,
                seed: int, rc_augment: bool, num_workers: int,
                shuffle: bool) -> tuple[DataLoader, DistributedSampler | None]:
    ds = WindowDataset(split, structural=False, rc_augment=rc_augment, seed=seed)
    sampler = None
    if world > 1:
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank,
                                     shuffle=shuffle, seed=seed, drop_last=shuffle)
    loader = DataLoader(
        ds, batch_size=batch_size, sampler=sampler,
        shuffle=(shuffle and sampler is None), collate_fn=collate,
        num_workers=num_workers, drop_last=shuffle, pin_memory=True,
    )
    return loader, sampler


def mask_tokens(tokens: torch.Tensor, mask_prob: float, gen: torch.Generator,
                p_mask: float, p_random: float) -> tuple[torch.Tensor, torch.Tensor]:
    """BERT-style masking restricted to ACGT. Returns (inputs, labels).

    Labels are -100 everywhere except the selected positions, so N and PAD can
    never contribute to the loss no matter what the model predicts there.
    """
    device = tokens.device
    candidates = (tokens >= NUC_LO) & (tokens <= NUC_HI)
    probs = torch.full(tokens.shape, mask_prob, device=device) * candidates
    selected = torch.bernoulli(probs, generator=gen).bool()

    labels = tokens.clone()
    labels[~selected] = -100

    inputs = tokens.clone()
    r = torch.rand(tokens.shape, device=device, generator=gen)
    inputs[selected & (r < p_mask)] = MASK_ID
    rand_hit = selected & (r >= p_mask) & (r < p_mask + p_random)
    if rand_hit.any():
        draw = torch.randint(NUC_LO, NUC_HI + 1, tokens.shape,
                             device=device, generator=gen)
        inputs[rand_hit] = draw[rand_hit]
    # the remaining selected positions keep their original token
    return inputs, labels


# --------------------------------------------------------------------------
# tau -- the Phase 4 gate
# --------------------------------------------------------------------------

class DeltaCapture:
    """Capture Delta as the model actually produces it, per layer and direction.

    Hooks `dt_proj`, whose output is `dt_pre`. For structural=False,
    Delta = softplus(dt_pre) exactly. The structural arm adds `W_dstruct(s)`
    before the softplus, so when Phase 4 reuses this the hook must move to after
    that addition or it will report the baseline's tau for the structural model.
    """

    def __init__(self, model: BiMambaLM):
        self.model = model
        self.handles = []
        self.store: dict[str, torch.Tensor] = {}

    def __enter__(self):
        for i, layer in enumerate(self.model.layers):
            for name in ("fwd", "rev"):
                d = getattr(layer, name)
                key = f"layer{i}.{name}"

                def hook(mod, inp, out, key=key):
                    self.store[key] = out.detach()
                self.handles.append(d.dt_proj.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()


@torch.no_grad()
def empirical_tau(model: BiMambaLM, batch_tokens: torch.Tensor,
                  n_positions: int = 128, seed: int = 0) -> dict:
    """tau = 1 / (Delta * |A|) in tokens, from Delta on real data.

    Sub-samples positions because the full tensor is
    (batch, length, d_inner, d_state) and would not fit.
    """
    core = unwrap(model)
    core.eval()
    with DeltaCapture(core) as cap:
        core(batch_tokens)
        g = torch.Generator(device=batch_tokens.device).manual_seed(seed)
        per_layer, all_tau = {}, []
        for i, layer in enumerate(core.layers):
            for name in ("fwd", "rev"):
                key = f"layer{i}.{name}"
                dt_pre = cap.store[key]                      # (b, l, d_inner)
                b, l, _ = dt_pre.shape
                idx = torch.randint(0, l, (min(n_positions, l),),
                                    device=dt_pre.device, generator=g)
                delta = F.softplus(dt_pre[:, idx, :]).float()      # (b, k, d_inner)
                absA = torch.exp(getattr(layer, name).A_log).float()  # (d_inner, n)
                tau = 1.0 / (delta.unsqueeze(-1) * absA.unsqueeze(0).unsqueeze(0))
                flat = tau.flatten()
                per_layer[key] = {
                    "tau_median": float(flat.median()),
                    "tau_p90": float(flat.quantile(0.90)),
                    "tau_p99": float(flat.quantile(0.99)),
                    "tau_max": float(flat.max()),
                    "frac_ge_5k": float((flat >= 5_000).float().mean()),
                    "frac_ge_100k": float((flat >= 100_000).float().mean()),
                }
                all_tau.append(flat[torch.randint(0, flat.numel(), (20000,),
                                                  device=flat.device, generator=g)])
    cat = torch.cat(all_tau)
    layer_max = [per_layer[f"layer{i}.{d}"]["tau_max"]
                 for i in range(len(core.layers)) for d in ("fwd", "rev")]
    fwd_max = [per_layer[f"layer{i}.fwd"]["tau_max"] for i in range(len(core.layers))]
    summary = {
        "tau_median": float(cat.median()),
        "tau_p90": float(cat.quantile(0.90)),
        "tau_p99": float(cat.quantile(0.99)),
        "tau_max": float(max(layer_max)),
        "frac_ge_5k": float((cat >= 5_000).float().mean()),
        "frac_ge_100k": float((cat >= 100_000).float().mean()),
        # relay heuristic, per architecture_spec.md 4.1.4 -- NOT a bound
        "relayed_sum_layer_tau_max_fwd": float(sum(fwd_max)),
    }
    core.train()
    return {"summary": summary, "per_layer": per_layer}


# --------------------------------------------------------------------------
# eval
# --------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, mask_prob, p_mask, p_random,
             eval_seed: int, world: int) -> dict:
    """Validation loss. The masking pattern is regenerated from a fixed seed at
    every eval so the number moves because the model moved, not the mask."""
    model.eval()
    gen = torch.Generator(device=device).manual_seed(eval_seed)
    tot_loss = torch.zeros((), device=device, dtype=torch.float64)
    tot_correct = torch.zeros((), device=device, dtype=torch.float64)
    tot_count = torch.zeros((), device=device, dtype=torch.float64)
    for batch in loader:
        tokens = batch["tokens"].to(device, non_blocking=True)
        inputs, labels = mask_tokens(tokens, mask_prob, gen, p_mask, p_random)
        logits = model(inputs)
        sel = labels != -100
        if not sel.any():
            continue
        lg, lb = logits[sel], labels[sel]
        loss = F.cross_entropy(lg, lb, reduction="sum")
        tot_loss += loss.double()
        tot_correct += (lg.argmax(-1) == lb).sum().double()
        tot_count += sel.sum().double()
    if world > 1:
        for t in (tot_loss, tot_correct, tot_count):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
    model.train()
    n = max(float(tot_count), 1.0)
    nats = float(tot_loss) / n
    return {
        "val_loss_nats": nats,
        "val_bits_per_nucleotide": nats / LN2,
        "val_accuracy": float(tot_correct) / n,
        "val_masked_positions": int(tot_count),
    }


# --------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------

def save_ckpt(path: Path, model, opt, sched, step: int, args, metrics: list,
              gen_state) -> None:
    core = unwrap(model)
    tmp = path.with_suffix(".tmp")
    torch.save({
        "step": step,
        "model": core.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": sched.state_dict(),
        "args": vars(args),
        "metrics": metrics,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "numpy_rng": np.random.get_state(),
        "train_gen": gen_state,
    }, tmp)
    tmp.replace(path)


def load_ckpt(path: Path, model, opt, sched, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    core = unwrap(model)
    core.load_state_dict(ck["model"])
    opt.load_state_dict(ck["optimizer"])
    sched.load_state_dict(ck["scheduler"])
    torch.set_rng_state(ck["torch_rng"].cpu())
    try:
        torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])
    except Exception:
        pass
    np.random.set_state(ck["numpy_rng"])
    return ck["step"], ck["metrics"], ck.get("train_gen")


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def param_groups(model: BiMambaLM, weight_decay: float):
    """No weight decay on norms, biases, A_log or D -- decaying A_log would pull
    the SSM timescales toward a fixed point that has nothing to do with the data,
    which is precisely the quantity the F4 gate measures."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith("A_log") or name.endswith(".D"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def lr_lambda(step: int, warmup: int, total: int, final_frac: float):
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    prog = (step - warmup) / max(total - warmup, 1)
    prog = min(max(prog, 0.0), 1.0)
    return final_frac + (1 - final_frac) * 0.5 * (1 + math.cos(math.pi * prog))


def worker(rank: int, world: int, args: argparse.Namespace) -> None:
    is_main = rank == 0
    backend = "n/a"
    if world > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(args.port))
        # NCCL calls nvmlInit_v2() during topology discovery. On a box where the
        # loaded kernel module and the userspace NVIDIA libraries disagree, that
        # fails and takes NCCL with it even though CUDA itself is fine. gloo
        # needs no NVML, so it is the working fallback here.
        # init_process_group("nccl") is lazy and succeeds even when NCCL cannot
        # work, so probing it by catching an exception there does not work --
        # the failure surfaces at the first collective, inside DDP's constructor.
        # NVML is the actual dependency, so probe that directly.
        backend = args.backend
        if backend == "auto":
            backend = "nccl" if nvml_works() else "gloo"
            if is_main and backend == "gloo":
                print("[rank0] NVML is not functional, so NCCL cannot initialise; "
                      "using gloo")
        dist.init_process_group(backend, rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    # Both arms must run the same scan or matched-compute is a fiction.
    use_scan("triton")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    cfg = ModelConfig(
        d_model=args.d_model, n_layer=args.n_layer, d_state=args.d_state,
        d_conv=args.d_conv, expand=args.expand, vocab_size=args.vocab_size,
        structural=False,
    )
    model = BiMambaLM(cfg).to(device)
    n_params = model.n_params()
    tau_init = model.tau_stats()

    net = CheckpointedLM(model) if args.grad_checkpoint else model
    ddp_model = DDP(net, device_ids=[rank]) if world > 1 else net

    opt = torch.optim.AdamW(param_groups(model, args.weight_decay),
                            lr=args.lr, betas=(args.beta1, args.beta2),
                            eps=args.eps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, args.warmup_steps, args.steps, args.lr_final_frac))

    train_loader, train_sampler = make_loader(
        "train", args.batch_size, rank, world, args.seed,
        rc_augment=args.rc_augment, num_workers=args.num_workers, shuffle=True)
    val_loader, _ = make_loader(
        "val", args.eval_batch_size, rank, world, args.seed,
        rc_augment=False, num_workers=args.num_workers, shuffle=False)

    run_dir = RESULTS / args.run_name
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)

    # ---- run_config.yaml is written BEFORE the first optimiser step ----
    if is_main:
        config = {
            "run_name": args.run_name,
            "phase": "3 -- sequence-only baseline",
            "written_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "STARTED -- no metrics yet",
            "hypothesis_arm": "baseline (structural=False)",
            "seed": args.seed,
            "hyperparameters": {
                k: getattr(args, k) for k in (
                    "steps", "batch_size", "grad_accum", "eval_batch_size",
                    "lr", "lr_final_frac", "warmup_steps", "weight_decay",
                    "beta1", "beta2", "eps", "grad_clip",
                    "mask_prob", "p_mask", "p_random",
                    "rc_augment", "num_workers", "eval_every", "tau_every",
                    "ckpt_every", "eval_seed", "num_workers",
                )
            },
            "effective_batch": {
                "windows_per_optimizer_step":
                    args.batch_size * args.grad_accum * world,
                "tokens_per_optimizer_step":
                    args.batch_size * args.grad_accum * world * args.window,
                "world_size": world,
                "ddp_backend": backend,
            },
            "model": {
                "class": "BiMambaLM",
                "structural": False,
                "n_parameters": n_params,
                **{k: getattr(cfg, k) for k in (
                    "d_model", "n_layer", "d_state", "d_conv", "expand",
                    "vocab_size")},
                "d_inner": cfg.d_inner,
                "dt_rank": cfg.dt_rank,
                # Delta init range. Recorded because it sets the memory horizon
                # (tau_max at init is exactly 1/dt_min) and therefore decides
                # the F4 gate. Runs with different values are not comparable,
                # and the Phase 3 baseline in results/baselines/ was trained at
                # dt_min=1e-3.
                **{k: getattr(cfg, k) for k in ("dt_min", "dt_max", "dt_floor")},
            },
            "scan_backend": {
                "backend": "triton (forced)",
                "why": "the structural arm cannot use mamba_ssm's fused scan, so "
                       "the baseline uses the custom kernel too; otherwise "
                       "matched-compute is a fiction",
                "validated_by": "scripts/validate_kernel.py, 34/34 checks passed",
                "bitwise_reproducible": False,
                "bitwise_note": "dBmat, dCmat and dp accumulate through "
                                "tl.atomic_add across d_inner programs; measured "
                                "relative spread ~1e-6 (fp32 rounding)",
            },
            "grad_checkpointing": args.grad_checkpoint,
            "precision": {
                "dtype": "float32",
                "autocast": False,
                "why": "the Triton scan was validated in fp32 only",
            },
            "masking": {
                "mask_prob": args.mask_prob,
                "p_mask": args.p_mask,
                "p_random": args.p_random,
                "p_keep": round(1.0 - args.p_mask - args.p_random, 6),
                "mask_token_id": MASK_ID,
                "maskable_tokens": "ACGT only (ids 2-5)",
                "excluded": "N (id 6) and PAD (id 0) are never masked and never "
                            "contribute loss; bits/nucleotide is therefore per "
                            "resolved nucleotide and the uniform floor is 2.000",
            },
            "data": {
                "source": "scripts/phase1_dataset.py WindowDataset",
                "window": args.window,
                "structural_features_supplied": False,
                "n_train_windows": len(train_loader.dataset),
                "n_val_windows": len(val_loader.dataset),
            },
            "git": git_state(),
            "environment": env_state(),
            "tau_at_init_bias_only": tau_init,
            "tau_measurement_note":
                "tau_stats() uses dt_proj.bias only, which is a proxy that is "
                "good at init and unreliable after training; empirical tau from "
                "Delta on real batches is logged in metrics.json and is the "
                "number the F4 gate should be read from",
        }
        (run_dir / "run_config.yaml").write_text(
            yaml.safe_dump(plain(config), sort_keys=False,
                           default_flow_style=False),
            encoding="utf-8")
        print(f"[rank0] wrote {run_dir / 'run_config.yaml'}")
        print(f"[rank0] parameters: {n_params:,}")

    if world > 1:
        dist.barrier()

    metrics: list[dict] = []
    start_step = 0
    ckpt_path = run_dir / "checkpoint.pt"
    if args.resume and ckpt_path.exists():
        start_step, metrics, _ = load_ckpt(ckpt_path, ddp_model, opt, sched, device)
        if is_main:
            print(f"[rank0] resumed from step {start_step}")

    train_gen = torch.Generator(device=device).manual_seed(args.seed * 100003 + 17)

    def append_metric(rec: dict) -> None:
        if not is_main:
            return
        metrics.append(rec)
        tmp = (run_dir / "metrics.json").with_suffix(".tmp")
        tmp.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        tmp.replace(run_dir / "metrics.json")

    ddp_model.train()
    step = start_step
    epoch = 0
    t_start = time.time()
    running, running_n = 0.0, 0
    data_iter = iter(train_loader)
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    while step < args.steps:
        opt.zero_grad(set_to_none=True)
        micro_loss = 0.0
        for _ in range(args.grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                epoch += 1
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch)
                data_iter = iter(train_loader)
                batch = next(data_iter)
            tokens = batch["tokens"].to(device, non_blocking=True)
            inputs, labels = mask_tokens(tokens, args.mask_prob, train_gen,
                                         args.p_mask, args.p_random)
            logits = ddp_model(inputs)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   labels.view(-1), ignore_index=-100)
            (loss / args.grad_accum).backward()
            micro_loss += float(loss) / args.grad_accum

        gnorm = torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), args.grad_clip)
        opt.step()
        sched.step()
        step += 1
        running += micro_loss
        running_n += 1

        if not math.isfinite(micro_loss):
            if is_main:
                print(f"[rank0] NON-FINITE LOSS at step {step}: {micro_loss}")
                append_metric({"step": step, "event": "non_finite_loss",
                               "train_loss_nats": micro_loss})
            raise SystemExit(3)

        if is_main and step % args.log_every == 0:
            avg = running / max(running_n, 1)
            el = time.time() - t_start
            print(f"[rank0] step {step:>6}/{args.steps}  "
                  f"loss {avg:.4f} nats  {avg/LN2:.4f} bits  "
                  f"lr {sched.get_last_lr()[0]:.2e}  gnorm {float(gnorm):.2f}  "
                  f"{el/step:.2f}s/step", flush=True)
            running, running_n = 0.0, 0

        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate(ddp_model, val_loader, device, args.mask_prob,
                          args.p_mask, args.p_random, args.eval_seed, world)
            rec = {"step": step, "epoch": epoch,
                   "train_loss_nats": micro_loss,
                   "train_bits_per_nucleotide": micro_loss / LN2,
                   "lr": sched.get_last_lr()[0],
                   "grad_norm": float(gnorm),
                   "elapsed_s": time.time() - t_start,
                   **ev}
            if step % args.tau_every == 0 or step == args.steps:
                probe = next(iter(val_loader))["tokens"][:1].to(device)
                core = unwrap(ddp_model)
                rec["tau_empirical"] = empirical_tau(core, probe, seed=args.eval_seed)
                rec["tau_bias_only"] = core.tau_stats()
            append_metric(rec)
            if is_main:
                print(f"[rank0] eval  step {step:>6}  "
                      f"val {ev['val_bits_per_nucleotide']:.4f} bits  "
                      f"acc {ev['val_accuracy']:.4f}", flush=True)
                if "tau_empirical" in rec:
                    s = rec["tau_empirical"]["summary"]
                    print(f"[rank0] tau   median {s['tau_median']:.1f}  "
                          f"p99 {s['tau_p99']:.1f}  max {s['tau_max']:.1f} tokens  "
                          f">=5k {s['frac_ge_5k']:.4f}  "
                          f">=100k {s['frac_ge_100k']:.6f}", flush=True)

        if is_main and (step % args.ckpt_every == 0 or step == args.steps):
            save_ckpt(ckpt_path, ddp_model, opt, sched, step, args, metrics,
                      train_gen.get_state())

    if is_main:
        cfg_path = run_dir / "run_config.yaml"
        doc = yaml.safe_load(cfg_path.read_text())
        doc["status"] = "COMPLETED"
        doc["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        doc["wall_clock_s"] = time.time() - t_start
        cfg_path.write_text(yaml.safe_dump(plain(doc), sort_keys=False,
                                           default_flow_style=False),
                            encoding="utf-8")
        print(f"[rank0] done in {time.time() - t_start:.0f}s")

    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch-size", type=int, default=2, help="windows per GPU")
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--eval-batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--lr-final-frac", type=float, default=0.1)
    p.add_argument("--warmup-steps", type=int, default=300)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--mask-prob", type=float, default=0.15)
    p.add_argument("--p-mask", type=float, default=0.8)
    p.add_argument("--p-random", type=float, default=0.1)
    p.add_argument("--rc-augment", action="store_true", default=True)
    p.add_argument("--no-rc-augment", dest="rc_augment", action="store_false")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layer", type=int, default=16)
    p.add_argument("--d-state", type=int, default=16)
    p.add_argument("--d-conv", type=int, default=4)
    p.add_argument("--expand", type=int, default=2)
    p.add_argument("--vocab-size", type=int, default=16)
    p.add_argument("--window", type=int, default=32768)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--tau-every", type=int, default=500)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-seed", type=int, default=1234)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    p.add_argument("--backend", choices=("auto", "nccl", "gloo"), default="auto")
    p.add_argument("--grad-checkpoint", action="store_true", default=False)
    p.add_argument("--port", type=int, default=29511)
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--smoke", action="store_true",
                   help="tiny config, few hundred steps, for wiring validation")
    args = p.parse_args()

    if args.smoke:
        args.d_model, args.n_layer, args.d_state = 64, 2, 16
        args.steps = args.steps if args.steps != 6000 else 300
        args.warmup_steps = 30
        args.eval_every = args.tau_every = args.ckpt_every = 100
        args.log_every = 25
        args.batch_size = args.eval_batch_size = 2
        args.grad_accum = 1
        args.run_name = args.run_name or f"smoke_seed{args.seed}"
    args.run_name = args.run_name or f"baseline_seed{args.seed}"
    return args


def main() -> int:
    args = parse()
    if not torch.cuda.is_available():
        print("No CUDA device.")
        return 2
    world = max(1, min(args.gpus, torch.cuda.device_count()))
    print(f"launching {world} process(es); run_name={args.run_name}")
    if world > 1:
        mp.spawn(worker, args=(world, args), nprocs=world, join=True)
    else:
        worker(0, 1, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
