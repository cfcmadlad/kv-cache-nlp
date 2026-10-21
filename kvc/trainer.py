import math
import os
import signal

import torch

from kvc.runlog import RunLog

PREEMPT_SIGNAL = getattr(signal, "SIGUSR1", None)


def lr_at(step: int, cfg: dict) -> float:
    if step < cfg["warmup"]:
        return cfg["lr"] * (step + 1) / cfg["warmup"]
    progress = (step - cfg["warmup"]) / max(1, cfg["steps"] - cfg["warmup"])
    floor = cfg.get("min_lr_frac", 0.1)
    return cfg["lr"] * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


def _to(t, device):
    return tuple(a.to(device) for a in t) if isinstance(t, tuple) else t.to(device)


def _save(path, model, opt, step) -> None:
    state = {"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
             "rng": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def fit(model, opt, batch_fn, cfg: dict, run: RunLog, device: str, stop_when=None) -> str:
    flag = {"stop": False}
    if PREEMPT_SIGNAL is not None:
        signal.signal(PREEMPT_SIGNAL, lambda *_: flag.update(stop=True))
    ckpt = run.dir / "ckpt.pt"
    step = 0
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=True)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        torch.set_rng_state(state["rng"].cpu())
        if "cuda_rng" in state and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
        step = state["step"]
        run.resumed_from(step)
        print(f"resumed at step {step}", flush=True)
    model.train()
    while step < cfg["steps"]:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        x, y = batch_fn(step)
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=cfg.get("bf16", False)):
            _, loss = model(_to(x, device), _to(y, device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.get("grad_clip", 0) > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        opt.step()
        step += 1
        if step % cfg["log_every"] == 0:
            run.log(step, loss=round(loss.item(), 5), lr=opt.param_groups[0]["lr"])
        halt = flag["stop"] or (stop_when is not None and stop_when(step))
        if halt or step % cfg["ckpt_every"] == 0 or step == cfg["steps"]:
            _save(ckpt, model, opt, step)
        if halt and step < cfg["steps"]:
            run.close("preempted", step)
            (run.dir / "PREEMPTED").touch()
            return "preempted"
    run.close("done", step)
    return "done"
