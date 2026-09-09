"""Measurement hooks for the performance campaign. Every entry point is a no-op
unless its env var is set, so the timed benchmark run pays nothing.

  PROBE=1 PROBE_OUT=<dir>/rec.json   correctness checkpoints via `probe` (one file per rank)
  TPA_STEPTIME_OUT=<dir>             per-step CUDA-event time, GPU/CPU memory  -> steptimes.rank<r>.json
  TPA_PROFILE_WINDOW=a,b             torch.profiler over steps a..b (1-based, inclusive)
  TPA_TRACE_DIR=<dir>                where the chrome trace goes                -> trace.rank<r>.json
"""
import os, time, json, itertools

_T0 = time.time()                       # import time ~ process start, for the startup cost
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
    record_lazy(f"noisepred_absmean_step{s}", lambda: noise_pred.detach().float().abs().mean().item(), rtol=2e-2)
    if noise_pred_audio is not None:
        record_lazy(f"noisepred_audio_absmean_step{s}", lambda: noise_pred_audio.detach().float().abs().mean().item(), rtol=2e-2)
    record_lazy(f"loss_step{s}", lambda: loss.detach().float().item(), rtol=2e-2)


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
            self._params = [p for p in accelerator.unwrap_model(model).parameters() if p.requires_grad]
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
        if probe_on():
            torch = self.torch
            grads = [p.grad for p in self._params if p.grad is not None]
            if grads:
                norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.float()) for g in grads]))
                record(f"gradnorm_step{self.step}", norm.item(), rtol=2e-2)
            record(f"n_grads_step{self.step}", len(grads))

    def after_step(self):
        torch = self.torch
        if probe_on():
            sq = sum((p.detach().float() ** 2).sum() for p in self._params)
            record(f"lora_sqsum_step{self.step}", sq.item(), rtol=1e-3)
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
                    record(f"final_absmean/{name}", p.detach().float().abs().mean().item(), rtol=1e-3)
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


def _rss_gb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 2**20
    except OSError:
        pass
    return None
