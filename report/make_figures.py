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

plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10"], "font.monospace": ["cmtt10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "axes.unicode_minus": False, "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
                     "pdf.fonttype": 42, "figure.dpi": 150, "savefig.pad_inches": 0.0})


def num(s: str) -> float:
    return float(s.replace(",", "").replace("+", "").replace("%", "").strip())


def interval(s: str) -> tuple:
    m = re.match(r"\s*([+-]?[\d.]+)\s*\[([+-]?[\d.]+),\s*([+-]?[\d.]+)\]", s)
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def tables(name: str) -> list:
    out, title, sub, header, rows, text = [], None, None, None, [], []
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
                out.append({"title": title, "sub": sub, "text": " ".join(text), "rows": rows})
                header, rows, text = None, [], []
            if line.startswith("### "):
                sub = line[4:].strip()
            elif line.startswith("## "):
                title = line[3:].strip()
                sub = None
                text = []
            elif line.strip():
                text.append(line.strip())
    if header is not None:
        out.append({"title": title, "sub": sub, "text": " ".join(text), "rows": rows})
    return out


def save(fig, name: str, outdir: Path):
    fig.savefig(outdir / name)
    plt.close(fig)
    print(f"wrote {outdir / name}")


def fig_frontier(outdir: Path):
    base = {t["title"].split()[0]: t["rows"] for t in tables("comparison.md")}
    combo = {t["title"].split()[0]: t["rows"] for t in tables("deploy_combo.md")}
    fig, axes = plt.subplots(1, 2, figsize=(6.87, 2.6), sharey=False)
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
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "fig_frontier.pdf", outdir)


def fig_scale(outdir: Path):
    rows = [r for t in tables("scale_and_tuning.md") if t["rows"] and "Steps" in t["rows"][0] for r in t["rows"]]
    rows = [r for r in rows if r["Model"].startswith("blt_wikitext103mix_") and r["Budget"] == "64"]
    fig, axes = plt.subplots(1, 2, figsize=(6.87, 2.6))
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
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "fig_scale.pdf", outdir)


def fig_stream(outdir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(6.87, 2.6))
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
    fig.legend(handles=handles, loc="lower center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "fig_stream.pdf", outdir)


def fig_memory(outdir: Path):
    mem = tables("deploy_memory.md")[0]["rows"]
    bench = tables("deploy_bench.md")[0]["rows"]
    fig, axes = plt.subplots(1, 3, figsize=(6.87, 2.6))
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
    ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, columnspacing=0.8, handlelength=1.2)
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
    axes[1].legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, columnspacing=0.8, handlelength=1.2)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.9, bottom=0.36, wspace=0.34)
    save(fig, "fig_memory.pdf", outdir)


def fig_attention(outdir: Path):
    names = {"TinyStories": ["attention_tinystories.md", "attention_blt_tinystories_entropy_s1.md", "attention_blt_tinystories_entropy_s2.md"],
             "WikiText-103": ["attention_wikitext.md", "attention_blt_wikitext103_entropy_s1.md", "attention_blt_wikitext103_entropy_s2.md"],
             "TinyStories, code mix": ["attention_tinystories_mix.md"],
             "WikiText-103, code mix": ["attention_wikitext_mix.md", "attention_blt_wikitext103mix_entropy_s1.md", "attention_blt_wikitext103mix_entropy_s2.md"]}
    fig, axes = plt.subplots(2, 2, figsize=(3.3, 3.3), sharey=True, sharex=True)
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


def count(rows: list, col: str) -> tuple:
    better = tie = worse = 0
    for r in rows:
        if col not in r or "[" not in r[col]:
            continue
        mean, lo, hi = interval(r[col])
        if lo >= 0 and mean > 0:
            worse += 1
        elif hi <= 0 and mean < 0:
            better += 1
        else:
            tie += 1
    return better, tie, worse


def fig_scoreboard(outdir: Path):
    seeds = [r for t in tables("seeds.md") for r in t["rows"] if r.get("Model", "").startswith("blt")]
    online = [r for t in tables("online_policies.md") for r in t["rows"] if r.get("Model", "").startswith("blt")]
    test = [r for t in tables("test_split.md") for r in t["rows"] if r.get("Model", "").startswith("blt")]
    rows = [("High entropy vs random", count(seeds, "entropy - random")),
            ("High entropy vs recency", count(seeds, "entropy - recency")),
            ("High vs low entropy", count(seeds, "entropy - entropy_inv")),
            ("High entropy vs random (test)", count(test, "entropy - random")),
            ("SnapKV-style vs recency", count(online, "snap on - recency")),
            ("H2O vs recency", count(online, "h2o on - recency"))]
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    labels = [r[0] for r in rows]
    ypos = list(range(len(rows)))[::-1]
    for y, (label, (b, t, w)) in zip(ypos, rows):
        total = b + t + w
        left = 0
        for n, color, name in ((b, "#0072B2", "policy better"), (t, "#BDBDBD", "no clear difference"), (w, "#D55E00", "policy worse")):
            ax.barh(y, n / total, left=left, color=color, edgecolor="white", height=0.62, label=name)
            if n:
                ax.text(left + n / total / 2, y, str(n), ha="center", va="center", color="white" if color != "#BDBDBD" else "black", fontsize=6.5)
            left += n / total
        ax.text(1.01, y, f"n={total}", va="center", fontsize=6)
    ax.set_yticks(ypos, labels)
    ax.set_xlim(0, 1.16)
    ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.set_xlabel("Share of paired comparisons")
    ax.grid(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#0072B2", "#BDBDBD", "#D55E00")]
    fig.legend(handles, ["policy better", "no clear difference", "policy worse"], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "fig_scoreboard.pdf", outdir)


def fig_terciles(outdir: Path):
    ts = [t for t in tables("error_analysis.md") if t["sub"] and t["sub"].startswith("BPB by entropy tercile")]
    fig, axes = plt.subplots(1, 2, figsize=(6.87, 2.6), sharey=False)
    for ax, t, name in zip(axes, ts, ("TinyStories, 32 patches", "WikiText-103, 64 patches")):
        rows = t["rows"]
        pols = [("recency", "recency"), ("random", "random"), ("entropy", "keep high entropy"), ("entropy_inv", "keep low entropy")]
        width = 0.2
        for k, (key, label) in enumerate(pols):
            ax.bar([i + (k - 1.5) * width for i in range(3)], [num(r[key]) for r in rows], width, color=COLORS[key], label=label)
        ax.set_xticks(range(3), [f"{r['Tercile'].replace(' third', '')}\n({r['Share of bytes']} of bytes)" for r in rows])
        ax.set_title(name)
        ax.set_xlabel("Entropy tercile of the patch holding the byte")
    axes[0].set_ylabel("Extra BPB over the full cache")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "fig_terciles.pdf", outdir)


def fig_kept(outdir: Path):
    kept = [t for t in tables("error_analysis.md") if t["sub"] and t["sub"].startswith("Which patches") and "Policy" in t["rows"][0] and "lowest third" in t["rows"][0]]
    length = [t for t in tables("error_analysis.md") if t["sub"] and t["sub"].startswith("Which patches") and "Mean kept patch length (bytes)" in t["rows"][0]]
    fig, axes = plt.subplots(1, 3, figsize=(6.87, 2.6))
    for ax, t, name in zip(axes[:2], kept, ("TinyStories, 32 patches", "WikiText-103, 64 patches")):
        rows = {r["Policy"]: r for r in t["rows"]}
        width = 0.25
        for k, key in enumerate(("recency", "entropy", "entropy_inv")):
            ax.bar([i + (k - 1) * width for i in range(3)], [num(rows[key][c]) for c in ("lowest third", "middle third", "highest third")], width, color=COLORS[key], label=LABELS[key])
        ax.set_xticks(range(3), ["low", "middle", "high"])
        ax.set_xlabel("Entropy tercile of the patch")
        ax.set_title(name, fontsize=7.5)
    axes[0].set_ylabel("Share of old patches kept (%)")
    ax = axes[2]
    names = ("TinyStories", "WikiText-103")
    width = 0.25
    for k, key in enumerate(("recency", "entropy", "entropy_inv")):
        ax.bar([i + (k - 1) * width for i in range(2)], [num({r["Policy"]: r for r in t["rows"]}[key]["Mean kept patch length (bytes)"]) for t in length], width, color=COLORS[key])
    ax.set_xticks(range(2), names)
    ax.set_ylabel("Mean length of kept patches (bytes)")
    ax.set_title("Patch length of what is kept", fontsize=7.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in ("recency", "entropy", "entropy_inv")]
    fig.legend(handles, [LABELS[k] for k in ("recency", "entropy", "entropy_inv")], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "fig_kept.pdf", outdir)


def fig_models(outdir: Path):
    gate = {r["Model"]: r for t in tables_md(REPO_ROOT / "docs" / "results_summary.md") for r in t if "Model" in r and "TinyStories" in r}
    scale = {r["Model"]: r for t in tables("scale_and_tuning.md") for r in t["rows"] if "Steps" in r and r["Budget"] == "64"}
    rows = [("KN\n4-gram", gate["Kneser-Ney byte 4-gram"], "#7F7F7F"),
            ("BPE\nTransf.", gate["BPE transformer 13.8M, reported"], "#E69F00"),
            ("BLT\nstride 4", gate["BLT 13.3M, fixed stride 4"], "#56B4E9"),
            ("BLT\nentropy", gate["BLT 13.3M, entropy patching (seed 0)"], "#0072B2")]
    fig, axes = plt.subplots(1, 2, figsize=(6.87, 2.4))
    for ax, key, name in zip(axes, ("TinyStories", "WikiText"), ("TinyStories val", "WikiText-103 val")):
        vals = [(label, num(r[key]), color) for label, r, color in rows]
        if key == "WikiText":
            vals.append(("BLT entropy\n30k steps", interval(scale["blt_wikitext103mix_entropy_x10"]["Full BPB [95% CI]"])[0], "#009E73"))
        ax.bar(range(len(vals)), [v[1] for v in vals], color=[v[2] for v in vals])
        for i, v in enumerate(vals):
            ax.text(i, v[1] + 0.03, f"{v[1]:.2f}", ha="center", fontsize=7)
        ax.set_xticks(range(len(vals)), [v[0] for v in vals], fontsize=6.8)
        ax.set_title(name)
    axes[0].set_ylabel("Bits per byte (lower is better)")
    fig.tight_layout()
    save(fig, "fig_models.pdf", outdir)


def tables_md(path: Path) -> list:
    out, header, rows = [], None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if header is None:
                header = cells
            elif not set(line.replace("|", "").strip()) <= set("-: "):
                rows.append(dict(zip(header, cells)))
        elif header is not None:
            out.append(rows)
            header, rows = None, []
    if header is not None:
        out.append(rows)
    return out


def fig_seeds(outdir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(6.87, 2.6))
    for ax, ds, title in zip(axes, ("tinystories", "wikitext103"), ("TinyStories", "WikiText-103")):
        sec = [t for t in tables("seeds.md") if t["title"] == ds][0]
        full = {}
        for r in sec["rows"]:
            if r.get("Model", "").startswith("blt"):
                full[r["Model"]] = num(r["Full-cache BPB [95% CI]"].split("[")[0])
        for x, (stem, label) in enumerate((("entropy", "entropy\npatching"), ("fixed", "fixed\nstride 4"), ("fixed5", "fixed\nstride 5"))):
            for seed in (0, 1, 2):
                name = f"blt_{ds}_{stem}" + (f"_s{seed}" if seed else "")
                marker = SEED_STYLE[seed][0]
                ax.scatter(x + (seed - 1) * 0.12, full[name], marker=marker, s=28, color="#0072B2" if stem == "entropy" else "#555555", zorder=3)
            ax.plot([x - 0.18, x + 0.18], [sum(full[f"blt_{ds}_{stem}" + (f"_s{s}" if s else "")] for s in (0, 1, 2)) / 3] * 2, color="#D55E00", lw=1.4, zorder=2)
        ax.set_xticks(range(3), ["entropy\npatching", "fixed\nstride 4", "fixed\nstride 5"])
        ax.set_title(title)
    axes[0].set_ylabel("Full-cache BPB (3,000 steps)")
    fig.legend([plt.Line2D([], [], marker=SEED_STYLE[s][0], ls="", color="#555555", label=f"seed {s}") for s in (0, 1, 2)] + [plt.Line2D([], [], color="#D55E00", lw=1.4, label="mean of 3 seeds")], ["seed 0", "seed 1", "seed 2", "mean of 3 seeds"], loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    save(fig, "fig_seeds.pdf", outdir)


def fig_public(outdir: Path):
    import json
    cfg = json.loads((REPO_ROOT / "configs" / "analysis" / "public_models.json").read_text(encoding="utf-8"))
    flops = json.loads((REPO_ROOT / "configs" / "analysis" / "flops.json").read_text(encoding="utf-8"))
    m, b = cfg["llama3_8b"], cfg["kv_bytes_per_element"]
    head_dim = m["model_dim"] // m["attention_heads"]
    per_token = 2 * m["layers"] * m["kv_heads"] * head_dim * b
    per_token_full_heads = 2 * m["layers"] * m["attention_heads"] * head_dim * b
    weights_gb = m["nominal_parameters"] * m["weight_bytes_per_parameter"] / 1e9
    ctx = cfg["context_axis_tokens"]
    fig, axes = plt.subplots(1, 2, figsize=(6.87, 2.4), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    ax.plot(ctx, [c * per_token / 1e9 for c in ctx], marker="o", ms=3.5, color="#0072B2", label=f"{m['label']} KV cache (grouped-query, {m['kv_heads']} KV heads)")
    ax.plot(ctx, [c * per_token_full_heads / 1e9 for c in ctx], ls=":", color="#0072B2", label="same, one KV head per attention head (hypothetical)")
    ax.plot(ctx, [4 * c * per_token / 1e9 for c in ctx], marker="s", ms=3.5, color="#D55E00", label="same model reading bytes: 4x positions (hypothetical)")
    ax.axhline(weights_gb, color="k", lw=0.9, ls="--")
    ax.text(ctx[0] * 1.1, weights_gb * 1.15, f"weights, fp16 (nominal 8B parameters): {weights_gb:.0f} GB", fontsize=6.5)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ctx, [f"{c // 1024}k" for c in ctx])
    ax.minorticks_off()
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("KV cache per sequence (GB, fp16)")
    ax.set_title("Why the cache matters at production scale", fontsize=7.5)
    ax.legend(frameon=False, fontsize=5.6, loc="lower right")
    ax = axes[1]
    d, L = 384, 6
    per_entry_ours = 2 * L * d * b
    bytes_text = 1000
    items = [(f"{m['label']}\n(at {cfg['assumed_bytes_per_token']:.0f} bytes/token)", bytes_text / cfg["assumed_bytes_per_token"] * per_token / 1e6, "#7F7F7F"),
             ("our byte-level\nTransformer", bytes_text * per_entry_ours / 1e6, "#D55E00"),
             ("our BPE\nTransformer", bytes_text / flops["bytes_per_token"] * per_entry_ours / 1e6, "#E69F00"),
             ("our BLT,\nfull cache", bytes_text / flops["bytes_per_patch"] * per_entry_ours / 1e6, "#0072B2"),
             ("our BLT,\n4-bit", bytes_text / flops["bytes_per_patch"] * per_entry_ours * 2448 / 9216 / 1e6, "#009E73")]
    ax.bar(range(len(items)), [i[1] for i in items], color=[i[2] for i in items])
    for k, i in enumerate(items):
        ax.text(k, i[1] * 1.08, f"{i[1]:.1f}", ha="center", fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(range(len(items)), [i[0] for i in items], fontsize=5.6)
    ax.set_ylabel("KV cache per 1,000 bytes of text (MB)")
    ax.set_title("Per byte of text (models differ 600x in size)", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig_public.pdf", outdir)


def fig_entries(outdir: Path):
    block = re.search(r"### tinystories: document 0.*?```\n(.*?)\n```", (TABLES / "error_analysis.md").read_text(encoding="utf-8"), re.S).group(1)
    parts = block.split("|")
    chars, spans, used = [], [], 0
    for p in parts:
        if used + len(p) > 46:
            break
        spans.append((used, len(p)))
        chars.extend(p)
        used += len(p)
    fig, ax = plt.subplots(figsize=(6.87, 1.7))
    for i, c in enumerate(chars):
        ax.add_patch(plt.Rectangle((i, 1.0), 0.92, 0.8, fc="#E6E6E6", ec="#7F7F7F", lw=0.5))
        ax.text(i + 0.46, 1.4, "_" if c == " " else c, ha="center", va="center", fontsize=6.5, family="monospace")
    for k, (start, n) in enumerate(spans):
        ax.add_patch(plt.Rectangle((start, 0.0), n - 0.08, 0.8, fc="#0072B2" if k % 2 == 0 else "#56B4E9", ec="white", lw=0.8))
        ax.text(start + n / 2 - 0.04, 0.4, str(k + 1), ha="center", va="center", fontsize=6.5, color="white")
    ax.text(-0.4, 1.4, "bytes", ha="right", va="center", fontsize=7)
    ax.text(-0.4, 0.4, "patches", ha="right", va="center", fontsize=7)
    ax.text(len(chars) / 2, 2.15, f"{len(chars)} bytes = {len(chars)} cache entries in a byte-level model, {len(spans)} entries when grouped into entropy patches", ha="center", fontsize=7.5)
    ax.set_xlim(-6, len(chars) + 0.5)
    ax.set_ylim(-0.1, 2.5)
    ax.axis("off")
    save(fig, "fig_entries.pdf", outdir)


def table_flops(outdir: Path):
    secs = tables("flops.md")
    ctxs = {t["title"]: t["rows"] for t in secs if t["title"] and t["title"].startswith("Context of")}
    pick = [k for k in ctxs if k.startswith("Context of 1,024") or k.startswith("Context of 4,096")]
    names = [r["Model"] for r in ctxs[pick[0]]]
    lines = ["\\begin{tabular}{@{}lrrrr@{}}", "\\toprule", " & \\multicolumn{2}{c}{1,024 bytes} & \\multicolumn{2}{c}{4,096 bytes}\\\\", "Model & MFLOPs & MB & MFLOPs & MB\\\\", "\\midrule"]
    for i, name in enumerate(names):
        cells = [name.replace("\u2020", "")]
        for k in pick:
            r = ctxs[k][i]
            cells += [r["MFLOPs per byte"], f"{num(r['KV cache (KB, fp16)']) / 1000:.1f}"]
        lines.append(" & ".join(cells).replace("+", "$+$") + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (outdir.parent / "flops_table.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {outdir.parent / 'flops_table.tex'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report/figures")
    args = ap.parse_args()
    outdir = REPO_ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    for fn in (fig_frontier, fig_scale, fig_stream, fig_memory, fig_attention, fig_scoreboard, fig_terciles, fig_kept, fig_models, fig_seeds, fig_public, fig_entries, table_flops):
        fn(outdir)


if __name__ == "__main__":
    main()
