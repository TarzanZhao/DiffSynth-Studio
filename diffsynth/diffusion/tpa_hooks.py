"""Measurement hooks for the performance campaign. Every entry point is a no-op
unless its env var is set, so the timed benchmark run pays nothing.

  PROBE=1 PROBE_OUT=<dir>/rec.json   correctness checkpoints via `probe` (one file per rank)
  TPA_STEPTIME_OUT=<dir>             per-step CUDA-event time, GPU/CPU memory  -> steptimes.rank<r>.json
  TPA_PROFILE_WINDOW=a,b             torch.profiler over steps a..b (1-based, inclusive)
  TPA_TRACE_DIR=<dir>                where the chrome trace goes                -> trace.rank<r>.json
"""
import os, time, json, itertools

_T0 = time.time()                       # import time ~ process start, for the startup cost

# Tolerances, from `probe derive` over 3 runs x 8 ranks of the unmodified code (bf16, FA2 backward atomics,
# bf16 LoRA params under AdamW): roughly 3x the largest spread seen per family.
# Gradients: the total LoRA grad norm is not a usable gate. From step 2 on it swings up to 8x between
# identical runs (bf16 params diverge by rounding). At step 1 (identical params) it is reproducible for one
# kernel set, but four correct attention kernels (FA2, cuDNN, torch flash, mem-efficient; pairwise 0.3% per
# call) give 0.041..0.096, with per-block cosine ~0 for blocks 0-28: the end-to-end bf16 backward loses the
# gradient signal after ~20 blocks. Blocks 40-49 are stable across kernels (cosine >= 0.987, norm within
# 2.7%), so `gradtail_step1` gates the backward pass and the totals are record-only (NaN / blow-up).
TOL = {
    "noisepred_absmean": dict(rtol=2e-2), "noisepred_audio_absmean": dict(rtol=2.5e-2),
    "loss": dict(rtol=0.15, atol=0.03), "lora_sqsum": dict(rtol=1.5e-2),
    "gradnorm": dict(rtol=20.0), "gradtail_step1": dict(rtol=0.10),
    "final_absmean_A": dict(rtol=2e-2), "final_absmean_B": dict(atol=4e-4),
}
_loss_step = itertools.count(1)

try:
    import probe as _probe
except ImportError:           # the recorder is optional; without it the checkpoints are silent
    _probe = None


def _rank():
    return int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))


def probe_on():
    return _probe is not None and _probe.enabled()


def record(tag, value, **tol):
    if _probe is not None:
        _probe.record(tag, value, **tol)


def record_lazy(tag, fn, **tol):
    if _probe is not None:
        _probe.record_lazy(tag, fn, **tol)


def loss_checkpoints(timestep_video, inputs, noise_pred, noise_pred_audio, loss):
    """Called once per step from FlowMatchSFTMiniMaxH3AudioVideoLoss. Boundaries: the sampled
    timestep, the shapes that enter the DiT, the DiT outputs, the loss."""
    if not probe_on():
        return
    s = next(_loss_step)
    record(f"timestep_step{s}", float(timestep_video.reshape(-1)[0].item()), rtol=0.0)
    if s == 1:
        for k in ("input_latents", "audio_input_latents", "prompt_embeds"):
            v = inputs.get(k)
            if hasattr(v, "shape"):
                record(f"shape_{k}", str(tuple(v.shape)))
    record_lazy(f"noisepred_absmean_step{s}", lambda: noise_pred.detach().float().abs().mean().item(), **TOL["noisepred_absmean"])
    if noise_pred_audio is not None:
        record_lazy(f"noisepred_audio_absmean_step{s}", lambda: noise_pred_audio.detach().float().abs().mean().item(), **TOL["noisepred_audio_absmean"])
    record_lazy(f"loss_step{s}", lambda: loss.detach().float().item(), **TOL["loss"])


class StepHooks:
    """Wraps the training loop in runner.py. Construct after accelerator.prepare."""

    def __init__(self, accelerator, model):
        import torch
        self.torch = torch
        self.accelerator = accelerator
        self.model = model
        self.rank = _rank()
        self.step = 0
        self.steptime_out = os.environ.get("TPA_STEPTIME_OUT")
        self.rows = []
        self._ev = None
        self.prof = None
        win = os.environ.get("TPA_PROFILE_WINDOW")
        if win:
            a, b = (int(x) for x in win.split(","))
            trace_dir = os.environ.get("TPA_TRACE_DIR", ".")
            os.makedirs(trace_dir, exist_ok=True)
            rank = self.rank

            def on_ready(p):
                p.export_chrome_trace(os.path.join(trace_dir, f"trace.rank{rank}.json"))

            self.prof = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(wait=max(a - 2, 0), warmup=1 if a >= 2 else 0, active=b - a + 1, repeat=1),
                on_trace_ready=on_ready, record_shapes=False, profile_memory=False, with_stack=False)
            self.prof.start()
        if self.steptime_out or probe_on():
            self._named_params = [(n, p) for n, p in accelerator.unwrap_model(model).named_parameters() if p.requires_grad]
            self._params = [p for _, p in self._named_params]
        self.t_ready = time.time()

    # -- per step -------------------------------------------------------------
    def step_begin(self):
        self.step += 1
        if self.steptime_out:
            torch = self.torch
            self._wall0 = time.time()
            self._ev = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            self._ev[0].record()

    def after_backward(self, loss):
        dump = os.environ.get("TPA_DUMP_GRADS")
        if dump and self.step == 1 and self.rank == 0:   # step-1 LoRA gradients, for offline comparison
            m = self.accelerator.unwrap_model(self.model)
            self.torch.save({n: p.grad.detach().float().cpu() for n, p in m.named_parameters() if p.grad is not None}, dump)
        if probe_on():
            torch = self.torch
            grads = [p.grad for p in self._params if p.grad is not None]
            if grads:
                norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.float()) for g in grads]))
                record(f"gradnorm_step{self.step}", norm.item(), **TOL["gradnorm"])
            record(f"n_grads_step{self.step}", len(grads))
            if self.step == 1:
                tail = [p.grad for n, p in self._named_params if p.grad is not None and _tail_block(n)]
                if tail:
                    tnorm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.float()) for g in tail]))
                    record("gradtail_step1", tnorm.item(), **TOL["gradtail_step1"])

    def after_step(self):
        torch = self.torch
        if probe_on():
            sq = sum((p.detach().float() ** 2).sum() for p in self._params)
            record(f"lora_sqsum_step{self.step}", sq.item(), **TOL["lora_sqsum"])
        if self.steptime_out:
            self._ev[1].record()
            self._ev[1].synchronize()
            ms = self._ev[0].elapsed_time(self._ev[1])
            self.rows.append({
                "step": self.step, "gpu_ms": ms, "wall_ms": (time.time() - self._wall0) * 1e3,
                "max_alloc_gb": torch.cuda.max_memory_allocated() / 2**30,
                "reserved_gb": torch.cuda.memory_reserved() / 2**30,
                "rss_gb": _rss_gb(),
            })
        if self.prof is not None:
            self.prof.step()

    # -- end of run -------------------------------------------------------------
    def end(self):
        if probe_on():
            m = self.accelerator.unwrap_model(self.model)
            for name, p in m.named_parameters():
                if p.requires_grad:
                    record(f"final_absmean/{name}", p.detach().float().abs().mean().item(), **TOL["final_absmean_B" if "lora_B" in name else "final_absmean_A"])
            _probe.flush()
        if self.prof is not None:
            self.prof.stop()
        if self.steptime_out:
            os.makedirs(self.steptime_out, exist_ok=True)
            out = {
                "rank": self.rank,
                "startup_s_to_loop": self.t_ready - _T0,
                "peak_alloc_gb": self.torch.cuda.max_memory_allocated() / 2**30,
                "peak_reserved_gb": self.torch.cuda.max_memory_reserved() / 2**30,
                "rss_gb_end": _rss_gb(),
                "steps": self.rows,
            }
            with open(os.path.join(self.steptime_out, f"steptimes.rank{self.rank}.json"), "w") as f:
                json.dump(out, f, indent=1)


def _tail_block(name, first=40):
    """LoRA tensors of DiT blocks >= `first` (the last 10 of 50): the part of the step-1 gradient that is
    stable across correct kernels (see TOL)."""
    if "token_refiner" in name or ".blocks." not in name:
        return False
    try:
        return int(name.split(".blocks.")[1].split(".")[0]) >= first
    except ValueError:
        return False


def _rss_gb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return None
