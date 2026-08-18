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
import zlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
# torch.utils.checkpoint is NOT auto-imported by `import torch` in this
# build; CheckpointedLM.forward calls it directly and raised AttributeError
# the first time --grad-checkpoint was ever exercised (2026-08-17).
import torch.utils.checkpoint  # noqa: F401
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
from phase1_dataset import (WindowDataset, PHI_CONTROLS,     # noqa: E402
                            PHI_CONTROL_SEED, S2_SHIFT_BP,
                            WINDOW as DATASET_WINDOW)

# One line each, written into run_config.yaml so a run states what it is without
# needing the spec open. Full definitions: architecture_spec.md 4.1.3.
PHI_CONTROL_DOC = {
    "none": "S0 -- real structure, phi as measured",
    "S1": "GLOBAL-PERM -- phi permuted uniformly across all bins; destroys "
          "sequence-structure correspondence AND local autocorrelation; "
          "preserves the marginal. The primary reliance probe.",
    "S2": f"CIRCULAR-SHIFT -- phi rolled by {S2_SHIFT_BP:,} bp; destroys "
          f"alignment only; preserves marginal and local autocorrelation.",
    "S3": "DISTANCE-MATCHED REWIRE -- phi recomputed from contacts resampled "
          "under the empirical P(s); removes locus-specific structure, keeps "
          "the 1D-distance-explainable component. The control that matters most.",
}

# Default only. Phase 3 baselines live here; Phase 4 passes --out-dir
# results/novel_model, per CLAUDE.md's deliverable layout. This was hardcoded
# until 2026-08-15, which meant a Phase 4 run wrote its results under
# results/baselines/ while its supervisor watched results/novel_model/ for the
# `status: COMPLETED` line -- the run would have trained correctly and then been
# retried until the attempt limit, because the supervisor could never see it
# finish.
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
        if c.c.structural:
            # Mirrors BiMambaLM.forward exactly. The encoder is 178 parameters
            # and runs once per forward, so it is deliberately NOT checkpointed
            # -- recomputing it would save nothing and s/s_rev are needed by
            # every layer anyway.
            if phi is None:
                raise ValueError("structural model called without phi")
            s = c.struct_encoder(phi)
            phi_rev = phi.flip(1)
            if symmetry is not None:
                phi_rev = phi_rev * symmetry.view(1, 1, -1).to(phi.dtype)
            s_rev = c.struct_encoder(phi_rev)
            if phi_valid is not None:
                m = phi_valid.unsqueeze(-1).to(s.dtype)
                s = s * m
                s_rev = s_rev * m.flip(1)
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
    def run(*a, strip=True):
        try:
            out = subprocess.run(a, cwd=REPO, capture_output=True, text=True,
                                 timeout=30).stdout
        except Exception:
            return "unavailable"
        # `git status --porcelain` encodes the status in the first two columns,
        # and an unstaged-only change leaves column 1 blank. A .strip() would
        # eat that leading space on the FIRST line only, so ln[3:] then cuts one
        # character too many and the run records "esults/..." instead of
        # "results/...". Every line after the first was fine, which is why this
        # went unnoticed until the v2 configs were diffed.
        return out.strip() if strip else out.rstrip("\n")
    dirty = run("git", "status", "--porcelain", strip=False)
    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(dirty),
        "dirty_files": ([ln[3:] for ln in dirty.splitlines()]
                        if dirty and dirty != "unavailable" else []),
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
    out = {
        "tokens": torch.from_numpy(
            np.stack([b["tokens"] for b in batch]).astype(np.int64)),
        "start": torch.tensor([b["start"] for b in batch], dtype=torch.int64),
    }
    # phi is present only when the dataset was built structural=True. The rc
    # augmentation returns reversed VIEWS (negative stride), which
    # torch.from_numpy rejects, hence the ascontiguousarray.
    if "phi" in batch[0]:
        out["phi"] = torch.from_numpy(np.ascontiguousarray(
            np.stack([b["phi"] for b in batch]), dtype=np.float32))
        out["phi_valid"] = torch.from_numpy(np.ascontiguousarray(
            np.stack([b["phi_valid"] for b in batch]).astype(np.bool_)))
    return out


def batch_struct(batch: dict, device, symmetry) -> dict:
    """The structural kwargs for BiMambaLM.forward, or {} for the baseline.

    Kept in one place so the training loop, the eval loop and the tau probe
    cannot drift apart: a structural model silently fed phi=None raises, but a
    structural model fed phi in training and not in eval would quietly report a
    validation number for a different model than the one being trained.
    """
    if "phi" not in batch:
        return {}
    return {
        "phi": batch["phi"].to(device, non_blocking=True),
        "phi_valid": batch["phi_valid"].to(device, non_blocking=True),
        "symmetry": symmetry,
    }


def make_loader(split: str, batch_size: int, rank: int, world: int,
                seed: int, rc_augment: bool, num_workers: int,
                shuffle: bool, structural: bool = False,
                phi_control: str = "none"
                ) -> tuple[DataLoader, DistributedSampler | None]:
    ds = WindowDataset(split, structural=structural, rc_augment=rc_augment,
                       seed=seed, phi_control=phi_control)
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
    Delta = softplus(dt_pre) exactly.

    The structural arm adds `W_dstruct(s)` to dt_pre BEFORE the softplus
    (`model.py::MambaDirection.forward`), so hooking dt_proj alone would report
    the BASELINE's tau for the structural model -- the exact confusion this
    class was warned about. `W_dstruct` is therefore hooked as well and the two
    outputs are summed, which reconstructs the real pre-softplus argument
    without touching the model. Both fire exactly once per direction per
    forward, so pairing them by key is safe.

    Reading tau for the structural arm from dt_proj alone would be a silent
    error, not a loud one: the numbers would look entirely plausible.
    """

    def __init__(self, model: BiMambaLM):
        self.model = model
        self.handles = []
        self.store: dict[str, torch.Tensor] = {}
        self.struct: dict[str, torch.Tensor] = {}

    def __enter__(self):
        for i, layer in enumerate(self.model.layers):
            for name in ("fwd", "rev"):
                d = getattr(layer, name)
                key = f"layer{i}.{name}"

                def hook(mod, inp, out, key=key):
                    self.store[key] = out.detach()
                self.handles.append(d.dt_proj.register_forward_hook(hook))

                if getattr(d, "W_dstruct", None) is not None:
                    def shook(mod, inp, out, key=key):
                        self.struct[key] = out.detach()
                    self.handles.append(d.W_dstruct.register_forward_hook(shook))
        return self

    def dt_pre(self, key: str) -> torch.Tensor:
        """The full pre-softplus argument, structural contribution included."""
        v = self.store[key]
        if key in self.struct:
            v = v + self.struct[key]
        return v

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles.clear()


@torch.no_grad()
def empirical_tau(model: BiMambaLM, batch_tokens: torch.Tensor,
                  n_positions: int = 128, seed: int = 0,
                  struct_kwargs: dict | None = None) -> dict:
    """tau = 1 / (Delta * |A|) in tokens, from Delta on real data.

    Sub-samples positions because the full tensor is
    (batch, length, d_inner, d_state) and would not fit.

    struct_kwargs carries phi/phi_valid/symmetry for the structural arm. It must
    be the SAME structural input the model trains on: Delta is input-dependent
    through W_dstruct, so probing a structural model with different phi measures
    a memory horizon the run never had.
    """
    core = unwrap(model)
    core.eval()
    with DeltaCapture(core) as cap:
        core(batch_tokens, **(struct_kwargs or {}))
        g = torch.Generator(device=batch_tokens.device).manual_seed(seed)
        per_layer, all_tau = {}, []
        for i, layer in enumerate(core.layers):
            for name in ("fwd", "rev"):
                key = f"layer{i}.{name}"
                dt_pre = cap.dt_pre(key)                      # (b, l, d_inner)
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
             eval_seed: int, world: int, symmetry=None) -> dict:
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
        logits = model(inputs, **batch_struct(batch, device, symmetry))
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
        structural=args.structural,
    )
    model = BiMambaLM(cfg).to(device)

    # PAIRED INITS (2026-08-17). torch.manual_seed above is not enough to
    # make the two arms share an init: the structural model instantiates
    # struct_encoder, W_dstruct and w_gate, drawing RNG the baseline never
    # draws, so every SHARED parameter constructed afterwards diverges. The
    # comparison was therefore unpaired -- reviewer weakness #3 -- which throws
    # away real statistical power for nothing.
    #
    # The fix is to build a BASELINE-config reference model from the same seed
    # and copy its parameters into whichever arm is actually training. The
    # reference is built identically in both processes -- same config, same
    # seed, same draw order -- so every shared tensor is bitwise equal across
    # arms by construction. Deriving a distribution per parameter instead does
    # NOT work: any statistic read off the already-constructed tensor is itself
    # arm-dependent, which was measured (90/275 tensors matched) before this
    # was replaced.
    #
    # Structural-only parameters keep their own init. W_dstruct is zero-init by
    # design and is untouched, so the arms remain numerically identical at
    # step 0 (the Phase 4 wiring gate depends on this).
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    _ref = BiMambaLM(ModelConfig(
        d_model=args.d_model, n_layer=args.n_layer, d_state=args.d_state,
        d_conv=args.d_conv, expand=args.expand, vocab_size=args.vocab_size,
        structural=False,
    ))
    _ref_sd = _ref.state_dict()
    with torch.no_grad():
        _shared = 0
        for name, prm in model.named_parameters():
            if name in _ref_sd and _ref_sd[name].shape == prm.shape:
                prm.copy_(_ref_sd[name].to(prm.device, prm.dtype))
                _shared += 1
    del _ref, _ref_sd

    # WARM START / STAGED PRETRAINING (2026-08-18). --init-from loads the
    # weights of a finished shorter-window run and trains on from step 0 at the
    # current window. A Mamba SSM has no positional embeddings, so every
    # parameter is window-independent by construction: the tensors transfer
    # unchanged and only the data pipeline changes.
    #
    # Weights ONLY. Optimiser state, LR schedule and step counter are not
    # restored -- this is a new run at a new width with its own warmup, not a
    # resume. --resume remains the mechanism for continuing an interrupted run
    # and the two are independent.
    #
    # This OVERWRITES the paired init above, and that is a real cost, not an
    # oversight: a warm-started pair is only as paired as the checkpoints it
    # starts from, and the 32,768 checkpoints predate the paired-init fix. A
    # run using --init-from therefore still carries reviewer weakness #3.
    # run_config.yaml records which of the two is in effect.
    if args.init_from:
        _src = Path(args.init_from)
        if not _src.exists():
            raise FileNotFoundError(f"--init-from: no checkpoint at {_src}")
        _ck = torch.load(_src, map_location=device, weights_only=False)
        _sd = _ck["model"]
        _loaded, _skipped = [], []
        with torch.no_grad():
            for name, prm in model.named_parameters():
                if name in _sd and _sd[name].shape == prm.shape:
                    prm.copy_(_sd[name].to(prm.device, prm.dtype))
                    _loaded.append(name)
                else:
                    _skipped.append(name)
        if not _loaded:
            raise RuntimeError(
                f"--init-from: {_src} shares no parameter with this model. "
                "Wrong arm or wrong architecture -- refusing to train from an "
                "init that silently did nothing.")
        if is_main:
            print(f"[rank0] WARM START from {_src} (source step "
                  f"{_ck.get('step')}): {len(_loaded)} tensors loaded, "
                  f"{len(_skipped)} left at init")
            if _skipped:
                print(f"[rank0]   left at init: {_skipped[:8]}"
                      f"{' ...' if len(_skipped) > 8 else ''}")
        _warm = {"init_from": str(_src), "source_step": _ck.get("step"),
                 "tensors_loaded": len(_loaded), "tensors_at_init": len(_skipped)}
        del _ck, _sd
    else:
        _warm = None

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
        rc_augment=args.rc_augment, num_workers=args.num_workers, shuffle=True,
        structural=args.structural, phi_control=args.phi_control)
    val_loader, _ = make_loader(
        "val", args.eval_batch_size, rank, world, args.seed,
        rc_augment=False, num_workers=args.num_workers, shuffle=False,
        structural=args.structural, phi_control=args.phi_control)

    # phi's antisymmetric coordinates (directionality index, the two directional
    # contact masses) must flip sign when the window is reversed for the reverse
    # pass -- failure mode F7. The dataset owns the definition; the model is
    # handed it as a tensor rather than re-deriving it, so the two can never
    # disagree about which coordinates are antisymmetric.
    symmetry = None
    if args.structural:
        symmetry = torch.from_numpy(
            np.asarray(train_loader.dataset.symmetry, dtype=np.float32)).to(device)

    run_dir = Path(args.out_dir) / args.run_name
    if is_main:
        run_dir.mkdir(parents=True, exist_ok=True)

    # ---- run_config.yaml is written BEFORE the first optimiser step ----
    if is_main:
        config = {
            "run_name": args.run_name,
            "phase": ("4 -- structural arm" if args.structural
                      else "3 -- sequence-only baseline"),
            "written_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "STARTED -- no metrics yet",
            "hypothesis_arm": (
                f"structural (structural=True, phi_control={args.phi_control})"
                if args.structural else "baseline (structural=False)"),
            "phi_control": args.phi_control,
            "phi_control_meaning": PHI_CONTROL_DOC[args.phi_control],
            "seed": args.seed,
            "init": ("warm start -- staged pretraining, NOT paired "
                     "(inherits the source checkpoint's init)" if _warm
                     else "fresh, paired against a baseline-config "
                          "reference model at the same seed"),
            "warm_start": _warm,
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
                "structural": args.structural,
                "n_parameters": n_params,
                **{k: getattr(cfg, k) for k in (
                    "d_model", "n_layer", "d_state", "d_conv", "expand",
                    "vocab_size", "d_struct", "d_struct_raw",
                    "d_struct_hidden", "use_permeability")},
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
                "structural_features_supplied": args.structural,
                "phi_control": args.phi_control,
                "phi_control_seed": (PHI_CONTROL_SEED if args.phi_control != "none"
                                     else None),
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
            logits = ddp_model(inputs, **batch_struct(batch, device, symmetry))
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
                          args.p_mask, args.p_random, args.eval_seed, world,
                          symmetry=symmetry)
            rec = {"step": step, "epoch": epoch,
                   "train_loss_nats": micro_loss,
                   "train_bits_per_nucleotide": micro_loss / LN2,
                   "lr": sched.get_last_lr()[0],
                   "grad_norm": float(gnorm),
                   "elapsed_s": time.time() - t_start,
                   **ev}
            if step % args.tau_every == 0 or step == args.steps:
                pb = next(iter(val_loader))
                probe = pb["tokens"][:1].to(device)
                # slice phi to the same single window, or Delta is computed from
                # a phi batch that does not match the tokens
                pb1 = {k: (v[:1] if torch.is_tensor(v) else v)
                       for k, v in pb.items()}
                core = unwrap(ddp_model)
                rec["tau_empirical"] = empirical_tau(
                    core, probe, seed=args.eval_seed,
                    struct_kwargs=batch_struct(pb1, device, symmetry))
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
            # RETENTION (2026-08-17). checkpoint.pt is overwritten every save,
            # so D1/D3 could only ever be read at the endpoint -- which is why
            # "the pathway is live only in layers 12-15" could not be separated
            # from "late layers simply converge first". Keeping a stamped copy
            # every args.keep_every steps gives those diagnostics a trajectory.
            if args.keep_every and step % args.keep_every == 0:
                save_ckpt(run_dir / f"checkpoint_step{step:06d}.pt", ddp_model,
                          opt, sched, step, args, metrics,
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
    p.add_argument("--window", type=int, default=DATASET_WINDOW,
                   help="reporting only; defaults to phase1_dataset.WINDOW "
                        "so run_config.yaml cannot state a stale width")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--tau-every", type=int, default=500)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--keep-every", type=int, default=1000,
                   help="also keep a step-stamped checkpoint every N "
                        "steps (0 disables); gives D1/D3 a trajectory")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--eval-seed", type=int, default=1234)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    p.add_argument("--backend", choices=("auto", "nccl", "gloo"), default="auto")
    p.add_argument("--grad-checkpoint", action="store_true", default=False)
    p.add_argument("--port", type=int, default=29511)
    p.add_argument("--init-from", type=str, default="",
                   help="warm start: load model weights from this "
                        "checkpoint and train from step 0. Weights only "
                        "-- not optimiser, schedule or step. Use for "
                        "staged pretraining across window widths; use "
                        "--resume to continue an interrupted run.")
    p.add_argument("--resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    p.add_argument("--out-dir", type=str, default=str(RESULTS),
                   help="directory the run directory is created under "
                        "(default results/baselines; Phase 4 uses "
                        "results/novel_model)")
    p.add_argument("--structural", action="store_true", default=False,
                   help="Phase 4 structural arm: condition Delta on phi")
    p.add_argument("--phi-control", choices=PHI_CONTROLS, default="none",
                   help="shuffled-structure control applied to phi before the "
                        "encoder (architecture_spec.md 4.1.3). Requires "
                        "--structural.")
    p.add_argument("--smoke", action="store_true",
                   help="tiny config, few hundred steps, for wiring validation")
    args = p.parse_args()

    if args.phi_control != "none" and not args.structural:
        p.error("--phi-control requires --structural: the baseline never reads "
                "phi, so a control on it would be a no-op and the run would be "
                "mislabelled as a control")

    if args.smoke:
        args.d_model, args.n_layer, args.d_state = 64, 2, 16
        args.steps = args.steps if args.steps != 6000 else 300
        args.warmup_steps = 30
        args.eval_every = args.tau_every = args.ckpt_every = 100
        args.log_every = 25
        args.batch_size = args.eval_batch_size = 2
        args.grad_accum = 1
        args.run_name = args.run_name or f"smoke_seed{args.seed}"
    if args.run_name is None:
        if args.structural:
            suffix = "" if args.phi_control == "none" else f"_{args.phi_control}"
            args.run_name = f"structural{suffix}_seed{args.seed}"
        else:
            args.run_name = f"baseline_seed{args.seed}"
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
