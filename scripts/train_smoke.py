import sys

import torch

from data_utils import REPO_ROOT
from kvc.config import load_config, run_id
from kvc.runlog import RunLog
from kvc.trainer import fit
from models.transformer import GPT


def make_corpus(seed: int, size: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    words = [torch.randint(97, 123, (int(torch.randint(2, 8, (1,), generator=g)),), generator=g) for _ in range(32)]
    order = torch.randint(0, len(words), (size,), generator=g)
    return torch.cat([torch.cat([words[i], torch.tensor([32])]) for i in order.tolist()])[:size]


def batcher(data: torch.Tensor, block: int, batch: int, seed: int):
    def get(step: int):
        g = torch.Generator().manual_seed(seed * 1_000_003 + step)
        idx = torch.randint(len(data) - block - 1, (batch, 1), generator=g) + torch.arange(block)
        return data[idx], data[idx + 1]
    return get


def build(cfg: dict, device: str):
    torch.manual_seed(cfg["seed"])
    m = cfg["model"]
    model = GPT(256, m["block"], m["n_layer"], m["n_head"], m["n_embd"], m["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    data = make_corpus(cfg["seed"], cfg["corpus_bytes"])
    return model, opt, batcher(data, m["block"], cfg["batch_size"], cfg["seed"])


def main():
    cfg_path = sys.argv[1]
    cfg = load_config(cfg_path, sys.argv[2:])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = RunLog(run_id(cfg_path), cfg, REPO_ROOT / "runs")
    (run.dir / "PREEMPTED").unlink(missing_ok=True)
    model, opt, batch_fn = build(cfg, device)
    status = fit(model, opt, batch_fn, cfg, run, device)
    if status == "done":
        x, y = batch_fn(10**9)
        with torch.no_grad():
            model.eval()
            loss = model(x.to(device), y.to(device))[1].item()
        run.close("done", cfg["steps"], {"final_loss": round(loss, 4), "params": model.num_params()})
    print(status, flush=True)


if __name__ == "__main__":
    main()
