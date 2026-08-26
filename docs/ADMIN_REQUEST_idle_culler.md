# Draft message to cluster admin — idle culler timeout

Status: DRAFT ONLY. Not sent. Send by hand once reviewed.

---

Subject: Request to raise/waive jupyterhub_idle_culler timeout for account
jupyter-238w1a5447

Hi,

I'd like to request that `jupyterhub_idle_culler --timeout=600` be raised or
waived for my account (jupyter-238w1a5447) on the shared GPU box (2x NVIDIA
L40S).

**The problem.** The culler kills the entire user cgroup — including
background training jobs — after 10 minutes with no browser activity in the
JupyterHub tab. It doesn't matter whether a training run is actively using the
GPU; only browser-tab activity counts, so `nohup`/`setsid`/reparenting to PID 1
give no protection.

**The concrete cost.** One phase of this project's model training needed about
nine hours of actual GPU compute. Because each culler kill required noticing,
restarting, and losing whatever checkpoint interval was in flight, that phase
took three calendar days rather than one. We've since added more frequent
checkpointing (every ~2 minutes of wall time) as a workaround, but that's a
mitigation for the symptom, not the cause, and it adds checkpoint I/O overhead
to every run.

**The request.** Either:
- raise the idle timeout for this account (e.g. to several hours), or
- waive it entirely while a training job is verified running (if there's a
  way to exempt a cgroup with active GPU utilization), or
- whatever mechanism you'd consider least disruptive to other users of the
  shared box.

Happy to provide more detail on the workload (fp32 Mamba-based sequence
model training, 2x L40S via gloo since NCCL can't initialize on this box due
to an unrelated NVML/kernel-module version mismatch) if useful for deciding.

Thanks,
[name]

---

## Context for whoever sends this (not part of the message itself)

- Referenced in `CLAUDE.md` §3 and §6, and `docs/RESEARCH_PLAN_2026-08-26.md`
  Phase B0(b): "the single highest-leverage fix available... Not yet
  requested" as of 2026-08-26.
- The "nine hours became three days" figure is the Phase 3 baseline run,
  recorded in the project's operational history (`CLAUDE.md` §3, §9).
- This is now more urgent than when first flagged: B0(a) (2026-08-26) found
  that 65,536 bp training needs `--grad-checkpoint` to fit in memory, which
  raises per-step time (~7.16 s/step measured on one GPU vs ~3.0 s/step at
  32,768 without it) — more wall-clock per checkpoint interval means more
  exposure to culler kills for the same amount of compute, not less.
