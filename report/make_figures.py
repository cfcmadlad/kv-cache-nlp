import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
TABLES = REPO_ROOT / "results" / "part3"
COLORS = {"recency": "#0072B2", "random": "#7F7F7F", "entropy": "#D55E00", "entropy_inv": "#009E73", "snap": "#CC79A7", "h2o": "#56B4E9", "uniform": "#000000", "tiers": "#E69F00"}
LABELS = {"recency": "recency", "random": "random", "entropy": "keep high entropy", "entropy_inv": "keep low entropy", "snap": "SnapKV-style", "h2o": "H2O", "uniform": "uniform bits", "tiers": "recent tiers"}
SEED_STYLE = {0: ("o", "-"), 1: ("s", "--"), 2: ("^", ":")}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "pdf.fonttype": 42, "figure.dpi": 150, "savefig.bbox": "tight"})


def num(s: str) -> float:
    return float(s.replace(",", "").replace("+", "").replace("%", "").strip())


def interval(s: str) -> tuple:
    m = re.match(r"\s*([+-]?[\d.]+)\s*\[([+-]?[\d.]+),\s*([+-]?[\d.]+)\]", s)
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def tables(name: str) -> list:
    out, title, header, rows, text = [], None, None, [], []
    for line in (TABLES / name).read_text(encoding="utf-8").splitlines():
        if line.startswith("|"):
            line = re.sub(r"(\w+)\|(\d+) ", r"\1@\2 ", line)
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if header is None:
                header = cells
            elif not set(line.replace("|", "").strip()) <= set("-: "):
                rows.append(dict(zip(header, cells)))
        else:
            if header is not None:
                out.append({"title": title, "text": " ".join(text), "rows": rows})
                header, rows, text = None, [], []
            if line.startswith("## "):
                title = line[3:].strip()
                text = []
            elif line.strip():
                text.append(line.strip())
    if header is not None:
        out.append({"title": title, "text": " ".join(text), "rows": rows})
    return out


def save(fig, name: str, outdir: Path):
    fig.savefig(outdir / name)
    plt.close(fig)
    print(f"wrote {outdir / name}")


def fig_frontier(outdir: Path):
    base = {t["title"].split()[0]: t["rows"] for t in tables("comparison.md")}
    combo = {t["title"].split()[0]: t["rows"] for t in tables("deploy_combo.md")}
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=False)
    ticks = {"tinystories": [100, 200, 400, 800], "wikitext103": [150, 300, 600, 1200, 2400]}
    for ax, ds, title in zip(axes, ("tinystories", "wikitext103"), ("TinyStories", "WikiText-103")):
        series = {}
        for r in base[ds]:
            s = r["Setting"].rstrip(" *")
            kb, cost = num(r["Mean KV cache (KB)"]), num(r["Cost over full"])
            m = re.match(r"evict to (\d+) patches, (\w+)", s)
            if m and m.group(2) in ("recency", "random", "entropy"):
                series.setdefault((m.group(2), "fp16"), []).append((kb, cost))
            m = re.match(r"uniform (\d+) bits", s)
            if m:
                series.setdefault(("uniform", "bits"), []).append((kb, cost))
        for r in combo[ds]:
            m = re.match(r"(\w+) (\d+) patches, 4-bit", r["Setting"].rstrip(" *"))
            if m:
                series.setdefault((m.group(1), "4bit"), []).append((num(r["Mean cache (KB)"]), num(r["Cost over full"])))
        for (name, kind), pts in series.items():
            pts.sort()
            xs, ys = zip(*pts)
            style = {"fp16": ("o", "-"), "bits": ("^", "-"), "4bit": ("s", "--")}[kind]
            ax.plot(xs, ys, marker=style[0], ls=style[1], color=COLORS[name], ms=3.5, lw=1.1, mfc=COLORS[name] if kind != "4bit" else "white")
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(ticks[ds], [f"{t:,}" for t in ticks[ds]])
        ax.set_xlabel("Mean global KV cache (KB, fp16 scale)")
        ax.set_title(title)
        ax.set_ylim(-0.002, 0.048 if ds == "tinystories" else 0.03)
    axes[0].set_ylabel("BPB cost over full cache")
    handles = [plt.Line2D([], [], color=COLORS[k], lw=1.5, label=LABELS[k]) for k in ("recency", "random", "entropy", "snap", "uniform")]
    handles += [plt.Line2D([], [], color="k", marker="o", ls="-", ms=3.5, label="eviction, fp16"), plt.Line2D([], [], color="k", marker="s", ls="--", mfc="white", ms=3.5, label="eviction + 4-bit")]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    save(fig, "fig_frontier.pdf", outdir)


def fig_scale(outdir: Path):
    rows = [r for t in tables("scale_and_tuning.md") if t["rows"] and "Steps" in t["rows"][0] for r in t["rows"]]
    rows = [r for r in rows if r["Model"].startswith("blt_wikitext103mix_") and r["Budget"] == "64"]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.4))
    groups = {}
    for r in rows:
        name = r["Model"]
        seed = int(re.search(r"_s(\d)$", name).group(1)) if re.search(r"_s\d$", name) else 0
        kind = "fixed" if "_fixed" in name else "entropy"
        groups.setdefault((kind, seed), []).append((int(r["Steps"].replace(",", "")), interval(r["entropy - random"]), interval(r["snap - recency"])))
    for ax, idx, title in zip(axes, (1, 2), ("Keep high entropy minus random", "SnapKV-style minus recency")):
        for (kind, seed), pts in sorted(groups.items()):
            pts.sort()
            xs = [p[0] for p in pts]
            ys = [p[idx][0] for p in pts]
            err = [[p[idx][0] - p[idx][1] for p in pts], [p[idx][2] - p[idx][0] for p in pts]]
            marker, ls = SEED_STYLE[seed]
            color = COLORS["entropy"] if kind == "entropy" else "#000000"
            label = f"entropy patching, seed {seed}" if kind == "entropy" else "fixed stride 4, seed 0"
            ax.errorbar(xs, ys, yerr=err, color=color, marker=marker, ls=ls if kind == "entropy" else "-.", ms=4, lw=1, capsize=2, label=label)
        ax.set_xscale("log")
        ax.set_xticks([3000, 30000], ["3,000", "30,000"])
        ax.minorticks_off()
        ax.set_xlabel("Training steps")
        ax.set_title(title + ", 64 patches")
        ax.axhline(0, color="k", lw=0.6)
    axes[0].set_ylabel("BPB difference (95% CI)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    save(fig, "fig_scale.pdf", outdir)


def fig_stream(outdir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6))
    ax = axes[0]
    acc = {}
    for t in tables("stream_16k.md"):
        if not t["rows"]:
            continue
        for r in t["rows"]:
            acc.setdefault((r["Policy"], int(r["Budget"])), []).append(num(r["Cost over chunked 4k full"]))
    for pol in ("recency", "snap", "random", "h2o", "entropy"):
        pts = sorted((b, sum(v) / len(v)) for (p, b), v in acc.items() if p == pol)
        ax.plot(*zip(*pts), marker="o", ms=3.5, lw=1.2, color=COLORS[pol], label=LABELS[pol])
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks([64, 128, 256], ["64", "128", "256"])
    ax.minorticks_off()
    ax.set_xlabel("Cache budget (patches)")
    ax.set_ylabel("BPB minus context reset (mean of 3 seeds)")
    ax.set_title("WikiText-103, 46 long documents")
    ax = axes[1]
    for t in tables("pg19_stream.md"):
        if not t["rows"]:
            continue
        steps = "30,000 steps" if "x10" in t["title"] else "3,000 steps"
        ls = "-" if "x10" in t["title"] else ":"
        full = re.search(r"reset4096 full cache within chunks: ([\d.]+); whole-book full cache: ([\d.]+)", t["text"])
        for pol in ("recency", "random", "entropy"):
            pts = sorted((int(r["Budget"]), interval(r["Policy - reset"])[0]) for r in t["rows"] if r["Policy"] == pol)
            ax.plot(*zip(*pts), marker="o", ms=3.5, lw=1.2, ls=ls, color=COLORS[pol])
        ax.axhline(float(full.group(2)) - float(full.group(1)), color="#000000", lw=0.8, ls=ls, alpha=0.6)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks([64, 128, 256], ["64", "128", "256"])
    ax.minorticks_off()
    ax.set_xlabel("Cache budget (patches)")
    ax.set_ylabel("BPB minus context reset")
    ax.set_title("PG-19, 20 books (black: whole-book full cache)")
    handles = [plt.Line2D([], [], color=COLORS[k], marker="o", ms=3.5, lw=1.2, label=LABELS[k]) for k in ("recency", "snap", "random", "h2o", "entropy")]
    handles += [plt.Line2D([], [], color="k", lw=1.2, ls="-", label="PG-19, 30,000-step model"), plt.Line2D([], [], color="k", lw=1.2, ls=":", label="PG-19, 3,000-step model")]
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    save(fig, "fig_stream.pdf", outdir)


def fig_memory(outdir: Path):
    mem = tables("deploy_memory.md")[0]["rows"]
    bench = tables("deploy_bench.md")[0]["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.4))
    ax = axes[0]
    ctx = [num(r["Context (bytes)"]) for r in mem]
    cols = [("Byte-level Transformer, full", "#D55E00", "byte-level Transformer"), ("BPE Transformer, full", "#E69F00", "BPE Transformer"), ("BLT patches, full fp16", "#0072B2", "BLT, full"),
            ("BLT, 256 patches fp16", "#009E73", "BLT, 256 patches"), ("BLT, 64 patches 4-bit", "#CC79A7", "BLT, 64 patches, 4-bit")]
    for col, color, label in cols:
        ax.plot(ctx, [num(r[col]) for r in mem], marker="o", ms=3, lw=1.2, color=color, label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Context (bytes)")
    ax.set_ylabel("KV cache per sequence (MB)")
    ax.set_title("Computed")
    ax.legend(frameon=False, fontsize=5.8, loc="upper left")
    ax = axes[1]
    series = {}
    for r in bench:
        series.setdefault(r["Policy"], []).append((num(r["Bytes decoded"]), num(r["Global cache (KB)"]) / 1000, num(r["ms per byte since previous row"])))
    for pol, pts in series.items():
        color = "#000000" if pol == "full" else {"64": "#CC79A7", "128": "#009E73", "256": "#0072B2"}[pol.split("@")[1]]
        axes[1].plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3, lw=1.2, color=color, label=pol.replace("@", " "))
        axes[2].plot([p[0] for p in pts], [p[2] for p in pts], marker="o", ms=3, lw=1.2, color=color, label=pol.replace("@", " "))
    for a, title, ylabel in ((axes[1], "Measured global cache", "MB"), (axes[2], "Measured decoding time", "ms per byte")):
        a.set_xscale("log", base=2)
        a.set_xlabel("Bytes decoded")
        a.set_ylabel(ylabel)
        a.set_title(title)
    axes[1].set_yscale("log")
    axes[1].legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout()
    save(fig, "fig_memory.pdf", outdir)


def fig_attention(outdir: Path):
    names = {"TinyStories": ["attention_tinystories.md", "attention_blt_tinystories_entropy_s1.md", "attention_blt_tinystories_entropy_s2.md"],
             "WikiText-103": ["attention_wikitext.md", "attention_blt_wikitext103_entropy_s1.md", "attention_blt_wikitext103_entropy_s2.md"],
             "TinyStories, code mix": ["attention_tinystories_mix.md"],
             "WikiText-103, code mix": ["attention_wikitext_mix.md", "attention_blt_wikitext103mix_entropy_s1.md", "attention_blt_wikitext103mix_entropy_s2.md"]}
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.3), sharey=True, sharex=True)
    for ax, (title, files) in zip(axes.flat, names.items()):
        for seed, f in enumerate(files):
            rows = tables(f)[0]["rows"]
            marker, ls = SEED_STYLE[seed]
            ax.plot([int(r["Layer"]) for r in rows], [num(r["Spearman(entropy, attention)"]) for r in rows], marker=marker, ls=ls, ms=3.5, lw=1.1, color=COLORS["entropy"], label=f"seed {seed}")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(title, fontsize=7.5)
    for ax in axes[1]:
        ax.set_xlabel("Global layer")
    for ax in axes[:, 0]:
        ax.set_ylabel("Spearman")
    axes[0, 0].legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_attention.pdf", outdir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report/figures")
    args = ap.parse_args()
    outdir = REPO_ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    for fn in (fig_frontier, fig_scale, fig_stream, fig_memory, fig_attention):
        fn(outdir)


if __name__ == "__main__":
    main()
