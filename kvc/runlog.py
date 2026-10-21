import json
import os
import platform
import socket
import subprocess
import time
from pathlib import Path

import torch


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    os.replace(tmp, path)


class RunLog:
    def __init__(self, run_id: str, config: dict, root="runs"):
        self.dir = Path(root) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.dir / "meta.json"
        self.start = time.time()
        _write_json(self.dir / "config.json", config)
        if self.meta_path.exists():
            self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))
        else:
            self.meta = {
                "id": run_id,
                "seed": config.get("seed"),
                "commit": _git("rev-parse", "HEAD"),
                "dirty": bool(_git("status", "--porcelain")),
                "host": socket.gethostname(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "slurm_job": os.environ.get("SLURM_JOB_ID"),
                "attempts": [],
                "wall_time_sec": 0.0,
                "metrics": {},
            }
        self.meta["status"] = "running"
        self.meta["attempts"].append({"started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                      "host": socket.gethostname(),
                                      "slurm_job": os.environ.get("SLURM_JOB_ID")})
        self._prior_wall = self.meta["wall_time_sec"]
        self._flush()

    def _flush(self) -> None:
        self.meta["wall_time_sec"] = round(self._prior_wall + time.time() - self.start, 2)
        _write_json(self.meta_path, self.meta)

    def log(self, step: int, **values) -> None:
        with open(self.dir / "log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({"step": step, **values}) + "\n")

    def resumed_from(self, step: int) -> None:
        self.meta["attempts"][-1]["resumed_from_step"] = step
        self._flush()

    def close(self, status: str, step: int, metrics: dict = None) -> None:
        self.meta["status"] = status
        self.meta["attempts"][-1]["ended_step"] = step
        if metrics:
            self.meta["metrics"].update(metrics)
        self._flush()
