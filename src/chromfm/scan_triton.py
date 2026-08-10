"""Triton selective scan carrying the permeability term p.

WHY THIS EXISTS
---------------
mamba_ssm's fused kernel computes Abar = exp(delta * A) and exposes `delta` as an
argument, so the Delta bias of architecture_spec.md 4.1 rides along for free.
It cannot express the permeability penalty, which needs

    Abar[t, d, n] = exp( delta[t, d] * A[d, n] - p[t] )

Folding p into delta would require delta' = delta + log(g) / A[d, n], which
depends on the state index n; a single per-channel delta cannot produce a decay
term uniform across n. So the term needs its own kernel.

    !!  VERIFICATION STATUS  !!
    This kernel has NEVER been executed. It was written on a CPU-only machine
    with no CUDA device and no Triton install. It is syntactically complete and
    the recurrence and its adjoint were derived by hand, but nothing here is
    numerically confirmed.

    Before any training run, execute scripts/validate_kernel.py on the GPU box.
    It checks forward values and every input gradient against the reference
    implementation in model.py. Do not trust a loss curve produced by an
    unvalidated scan -- a subtly wrong gradient trains to a plausible-looking
    number and quietly invalidates the experiment.

DESIGN
------
One program per (batch, channel) pair; the state h of size d_state lives in
registers while the program walks the sequence. The forward checkpoints h at
chunk boundaries into `states`; the backward walks chunks in reverse, recomputes
the forward inside each chunk from its checkpoint into a scratch buffer, then
runs the adjoint recurrence over that chunk. Memory is O(B * D * L/CHUNK * N)
for checkpoints rather than O(B * D * L * N) for the full trajectory.

ADJOINT (derived by hand; the thing validate_kernel.py is checking)
    dAbar[n] = dh[n] * h_prev[n]
    dh_prev[n] += dh[n] * Abar[n]
    ddelta    += sum_n dh[n] * h_prev[n] * Abar[n] * A[n]      (through Abar)
               + sum_n dh[n] * u * Bmat[n]                      (through Bbar)
    dA[n]     += dh[n] * h_prev[n] * Abar[n] * delta
    dp        += -sum_n dh[n] * h_prev[n] * Abar[n]
    dBmat[n]  += dh[n] * u * delta
    dCmat[n]  += dy * h[n]
    du        += dy * Dskip + sum_n dh[n] * delta * Bmat[n]
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    HAVE_TRITON = True
except Exception:                                     # CPU box, no GPU
    HAVE_TRITON = False
    triton = None

    class _Stub:
        @staticmethod
        def jit(fn):
            return fn

        @staticmethod
        def constexpr(x):
            return x
    tl = _Stub()


CHUNK = 64


if HAVE_TRITON:

    @triton.jit
    def _fwd_kernel(
        u_ptr, delta_ptr, A_ptr, Bm_ptr, Cm_ptr, Dskip_ptr, p_ptr,
        y_ptr, states_ptr,
        D: tl.constexpr, L, NCHUNK,
        su_b, su_d, su_l,
        sA_d, sA_n,
        sm_b, sm_n, sm_l,
        sp_b, sp_l,
        ss_b, ss_d, ss_c, ss_n,
        N: tl.constexpr, HAS_P: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // D
        d = pid % D
        n = tl.arange(0, N)

        A = tl.load(A_ptr + d * sA_d + n * sA_n)
        Dskip = tl.load(Dskip_ptr + d)
        h = tl.zeros([N], dtype=tl.float32)

        ub = u_ptr + b * su_b + d * su_d
        db = delta_ptr + b * su_b + d * su_d
        yb = y_ptr + b * su_b + d * su_d
        Bb = Bm_ptr + b * sm_b
        Cb = Cm_ptr + b * sm_b
        sb = states_ptr + b * ss_b + d * ss_d

        for t in range(L):
            if t % CHUNK == 0:
                tl.store(sb + (t // CHUNK) * ss_c + n * ss_n, h)

            dt = tl.load(db + t * su_l)
            ut = tl.load(ub + t * su_l)
            Bt = tl.load(Bb + n * sm_n + t * sm_l)
            Ct = tl.load(Cb + n * sm_n + t * sm_l)

            logdec = dt * A
            if HAS_P:
                logdec = logdec - tl.load(p_ptr + b * sp_b + t * sp_l)

            h = tl.exp(logdec) * h + dt * Bt * ut
            tl.store(yb + t * su_l, tl.sum(h * Ct, axis=0) + Dskip * ut)

    @triton.jit
    def _bwd_kernel(
        u_ptr, delta_ptr, A_ptr, Bm_ptr, Cm_ptr, Dskip_ptr, p_ptr,
        dy_ptr, states_ptr, hbuf_ptr,
        du_ptr, ddelta_ptr, dA_ptr, dBm_ptr, dCm_ptr, dDskip_ptr, dp_ptr,
        D: tl.constexpr, L, NCHUNK,
        su_b, su_d, su_l,
        sA_d, sA_n,
        sm_b, sm_n, sm_l,
        sp_b, sp_l,
        ss_b, ss_d, ss_c, ss_n,
        sh_b, sh_d, sh_t, sh_n,
        N: tl.constexpr, HAS_P: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // D
        d = pid % D
        n = tl.arange(0, N)

        A = tl.load(A_ptr + d * sA_d + n * sA_n)
        Dskip = tl.load(Dskip_ptr + d)

        ub = u_ptr + b * su_b + d * su_d
        db = delta_ptr + b * su_b + d * su_d
        gb = dy_ptr + b * su_b + d * su_d
        Bb = Bm_ptr + b * sm_b
        Cb = Cm_ptr + b * sm_b
        sb = states_ptr + b * ss_b + d * ss_d
        hb = hbuf_ptr + b * sh_b + d * sh_d

        dh = tl.zeros([N], dtype=tl.float32)
        dDskip_acc = tl.zeros([1], dtype=tl.float32)
        dA_acc = tl.zeros([N], dtype=tl.float32)

        for c in range(NCHUNK - 1, -1, -1):
            t0 = c * CHUNK
            tmax = tl.minimum(CHUNK, L - t0)

            # replay the forward inside this chunk from its checkpoint
            h = tl.load(sb + c * ss_c + n * ss_n)
            tl.store(hb + 0 * sh_t + n * sh_n, h)
            for i in range(tmax):
                t = t0 + i
                dt = tl.load(db + t * su_l)
                ut = tl.load(ub + t * su_l)
                Bt = tl.load(Bb + n * sm_n + t * sm_l)
                logdec = dt * A
                if HAS_P:
                    logdec = logdec - tl.load(p_ptr + b * sp_b + t * sp_l)
                h = tl.exp(logdec) * h + dt * Bt * ut
                tl.store(hb + (i + 1) * sh_t + n * sh_n, h)

            # adjoint pass, backwards through the same chunk
            for i in range(tmax - 1, -1, -1):
                t = t0 + i
                dt = tl.load(db + t * su_l)
                ut = tl.load(ub + t * su_l)
                Bt = tl.load(Bb + n * sm_n + t * sm_l)
                Ct = tl.load(Cb + n * sm_n + t * sm_l)
                dyt = tl.load(gb + t * su_l)

                h_t = tl.load(hb + (i + 1) * sh_t + n * sh_n)
                h_prev = tl.load(hb + i * sh_t + n * sh_n)

                logdec = dt * A
                if HAS_P:
                    logdec = logdec - tl.load(p_ptr + b * sp_b + t * sp_l)
                Abar = tl.exp(logdec)

                tl.atomic_add(dCm_ptr + b * sm_b + n * sm_n + t * sm_l, dyt * h_t)
                dh = dh + dyt * Ct

                common = dh * h_prev * Abar
                ddelta_t = tl.sum(common * A, axis=0) + tl.sum(dh * ut * Bt, axis=0)
                dA_acc += common * dt
                if HAS_P:
                    tl.atomic_add(dp_ptr + b * sp_b + t * sp_l, -tl.sum(common, axis=0))
                tl.atomic_add(dBm_ptr + b * sm_b + n * sm_n + t * sm_l, dh * ut * dt)

                du_t = dyt * Dskip + tl.sum(dh * dt * Bt, axis=0)
                tl.store(du_ptr + b * su_b + d * su_d + t * su_l, du_t)
                tl.store(ddelta_ptr + b * su_b + d * su_d + t * su_l, ddelta_t)
                dDskip_acc += dyt * ut

                dh = dh * Abar

        tl.atomic_add(dA_ptr + d * sA_d + n * sA_n, dA_acc)
        tl.atomic_add(dDskip_ptr + d, tl.sum(dDskip_acc, axis=0))


class _SelectiveScanTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, delta, A, Bm, Cm, Dskip, p):
        Bsz, D, L = u.shape
        N = A.shape[1]
        nchunk = (L + CHUNK - 1) // CHUNK
        u, delta, Bm, Cm = (x.contiguous() for x in (u, delta, Bm, Cm))
        A, Dskip = A.contiguous(), Dskip.contiguous()
        has_p = p is not None
        if has_p:
            p = p.contiguous()

        y = torch.empty_like(u)
        states = torch.empty((Bsz, D, nchunk, N), device=u.device, dtype=torch.float32)

        _fwd_kernel[(Bsz * D,)](
            u, delta, A, Bm, Cm, Dskip, p if has_p else u,
            y, states,
            D, L, nchunk,
            *u.stride(), *A.stride(), *Bm.stride(),
            *(p.stride() if has_p else (0, 0)),
            *states.stride(),
            N=N, HAS_P=has_p,
        )
        ctx.save_for_backward(u, delta, A, Bm, Cm, Dskip, p if has_p else None, states)
        ctx.has_p = has_p
        return y

    @staticmethod
    def backward(ctx, dy):
        u, delta, A, Bm, Cm, Dskip, p, states = ctx.saved_tensors
        Bsz, D, L = u.shape
        N = A.shape[1]
        nchunk = states.shape[2]
        has_p = ctx.has_p
        dy = dy.contiguous()

        du = torch.zeros_like(u)
        ddelta = torch.zeros_like(delta)
        dA = torch.zeros_like(A)
        dBm = torch.zeros_like(Bm)
        dCm = torch.zeros_like(Cm)
        dDskip = torch.zeros_like(Dskip)
        dp = torch.zeros_like(p) if has_p else torch.zeros((Bsz, L), device=u.device)
        hbuf = torch.empty((Bsz, D, CHUNK + 1, N), device=u.device, dtype=torch.float32)

        _bwd_kernel[(Bsz * D,)](
            u, delta, A, Bm, Cm, Dskip, p if has_p else u,
            dy, states, hbuf,
            du, ddelta, dA, dBm, dCm, dDskip, dp,
            D, L, nchunk,
            *u.stride(), *A.stride(), *Bm.stride(),
            *(p.stride() if has_p else (0, 0)),
            *states.stride(), *hbuf.stride(),
            N=N, HAS_P=has_p,
        )
        return du, ddelta, dA, dBm, dCm, dDskip, (dp if has_p else None)


def selective_scan_triton(u, delta, A, Bm, Cm, Dskip, p=None):
    """Drop-in replacement for selective_scan_ref with identical signature."""
    if not HAVE_TRITON or not u.is_cuda:
        raise RuntimeError(
            "selective_scan_triton needs a CUDA tensor and a Triton install; "
            "the reference scan in chromfm.model is the CPU path."
        )
    return _SelectiveScanTriton.apply(u, delta, A, Bm, Cm, Dskip, p)


def available(device: torch.device | str | None = None) -> bool:
    if not HAVE_TRITON:
        return False
    if device is None:
        return torch.cuda.is_available()
    return torch.device(device).type == "cuda"
