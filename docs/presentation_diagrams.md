# Diagrams for the talk (Mermaid)

Diagrams in aid of explanation at the viva and the presentation. They render on GitHub and in any Markdown viewer that speaks Mermaid; PNG copies for the slides live in `report/presentation/mermaid/` (rebuilt by `python scripts/render_mermaid.py`, which needs `pip install playwright` and a Chromium, passed with `--chromium`). The figures are transcribed from `docs/results_summary.md` and `results/part3/`.

## 1. The story in one picture

```mermaid
flowchart LR
    A["Byte-level LMs: no tokenizer,<br/>but 4x more positions"] --> B["KV cache grows 4x faster<br/>(deployment bottleneck)"]
    B --> C["BLT: one cache entry per patch<br/>(4.5 bytes)"]
    C --> D["Idea: patch entropy is already computed.<br/>Keep high-entropy patches?"]
    D --> E{"Tested at equal cache bytes,<br/>3 seeds, test split"}
    E -->|"Entropy rule"| F["Negative: worse than random in 68/72,<br/>worse than recency in 70/72"]
    E -->|"Attention rule + 4-bit + bounded cache"| G["Works: 7.4% of cache for +0.0012 BPB,<br/>SnapKV-style beats recency 24/24"]
```

## 2. System pipeline

```mermaid
flowchart LR
    X["raw UTF-8 bytes"] --> EM["entropy model<br/>3.3M params, causal"]
    EM --> P["patcher:<br/>new patch where entropy > threshold<br/>(4.5 bytes per patch)"]
    X --> LE["local encoder<br/>1 layer, width 256"]
    LE --> POOL["mean pool per patch<br/>linear to width 384"]
    P --> POOL
    POOL --> GT["global Transformer<br/>6 layers, width 384<br/>over patches"]
    GT --> KV[("KV cache:<br/>1 entry per patch<br/>9,216 bytes fp16")]
    KV --> LD["local decoder<br/>2 layers, width 256"]
    LE --> LD
    LD --> Y["next byte<br/>(256-way)"]
    POL{{"cache policy:<br/>recency / random / entropy /<br/>H2O / SnapKV-style<br/>+ 4-bit quantization"}} -.-> KV
```

## 3. What a cache policy does

```mermaid
flowchart TD
    S["new patch arrives"] --> Q{"cache over budget?"}
    Q -->|no| K["append entry"]
    Q -->|yes| R["rank old patches<br/>(first patch and 8 newest protected)"]
    R --> R1["recency: drop oldest"]
    R --> R2["random: drop random"]
    R --> R3["entropy: drop lowest-entropy"]
    R --> R4["H2O: drop least accumulated attention"]
    R --> R5["SnapKV-style: drop least attended<br/>by the last 16 patches"]
    R1 --> K
    R2 --> K
    R3 --> K
    R4 --> K
    R5 --> K
    K --> Z["optional: store kept entries in 4-bit"]
```

## 4. Experimental design

```mermaid
flowchart TB
    subgraph Data
        D1["TinyStories"] --- D2["WikiText-103"] --- D3["The Stack Smol (15% mix)"] --- D4["PG-19 (streaming)"]
    end
    subgraph Models
        M1["Kneser-Ney 4-gram<br/>(classical)"] --- M2["BPE Transformer 13.8M<br/>(modern baseline)"] --- M3["BLT-style 13.3M<br/>entropy or fixed patches"]
    end
    subgraph Evaluation
        E1["bits per byte, same documents"] --- E2["paired bootstrap, 3 seeds"] --- E3["test split once"] --- E4["memory, time, FLOPs"]
    end
    Data --> Models --> Evaluation
    Evaluation --> ERR["error analysis:<br/>entropy terciles, byte classes,<br/>kept patches, generations"]
```

## 5. Decision tree of the results

```mermaid
flowchart TD
    Q{"Is next-byte entropy a good<br/>keep signal?"} -->|"72 paired comparisons"| N["No"]
    N --> N1["worse than random 68, tie 4, better 0"]
    N --> N2["worse than recency 70, tie 2, better 0"]
    N --> N3["penalty grows 10x with 10x training"]
    N --> N4["keeps 1-2 byte fragments (98-99.7%)"]
    Q -->|"so what does work?"| Q2{"Alternatives"}
    Q2 --> W1["SnapKV-style (online):<br/>beats recency 24/24"]
    Q2 --> W2["eviction + 4-bit:<br/>7.4% of cache, +0.0012 BPB"]
    Q2 --> W3["bounded recency cache:<br/>beats context reset, constant memory"]
    Q2 --> W4["BLT: 2.8x fewer FLOPs/byte<br/>than byte-level Transformer"]
```

## 6. Timeline of the project

```mermaid
gantt
    title CS F429 project milestones (2026)
    dateFormat YYYY-MM-DD
    axisFormat %d %b
    section Part 1
    Proposal submitted (27 Aug)        :milestone, m1, 2026-08-27, 0d
    section Part 2
    Part 2 report and demo (15 Oct)    :milestone, m2, 2026-10-15, 0d
    section Part 3
    Byte model, policies, all experiments :done, p3a, 2026-10-21, 2026-10-25
    Report, error analysis, demo prep  :active, p3b, 2026-10-26, 2026-11-12
    Team target (12 Nov)               :milestone, m3, 2026-11-12, 0d
    Final submission (21 Nov)          :milestone, m4, 2026-11-21, 0d
```

## 7. Outcome of the main test (keep high entropy vs random)

```mermaid
pie showData
    title 72 paired comparisons: keep high entropy vs random
    "worse (interval above zero)" : 68
    "no clear difference" : 4
    "better" : 0
```

## 8. Demo flow

```mermaid
flowchart LR
    S(["python scripts/demo.py"]) --> O6["6: train a 1.8M-parameter<br/>byte model live (about 80 s)"]
    O6 --> O4["4: compress the KV cache of<br/>any text (a few seconds)"]
    O4 --> O7["7: stream 4,000 bytes with a<br/>real fixed-size cache (about 36 s)"]
    O7 --> O5["5: results tables"]
    O5 --> Q(["questions"])
```

## 9. KV cache per 1,000 bytes of text (MB, fp16)

```mermaid
xychart-beta
    title "KV cache per 1,000 bytes of text (MB)"
    x-axis ["Llama 3 8B*", "byte Transf.", "BPE Transf.", "BLT full", "BLT 4-bit"]
    y-axis "MB" 0 --> 35
    bar [32.8, 9.2, 2.3, 2.0, 0.5]
```

\* Llama 3 8B at an assumed 4 bytes per token; about 600 times more parameters than our models, so only the cost structure is comparable (`report/figures/fig_public.pdf`, `configs/public_models.json`).
