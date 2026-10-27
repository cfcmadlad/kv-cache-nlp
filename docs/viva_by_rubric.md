# Questions the examiner may put, by assessed item

The likely questions for each Part 3 rubric item (Novelty 3, Improvement 3, Results and error analysis 2, Challenges 2, Demonstration and viva 20), with brief answers and the file that backs each claim. The collation script (`python scripts/collate_viva.py`) folds this document into `docs/viva_notes.md`.

## Viva notes

### An opening statement (sixty seconds)

Byte-level language models need about four times the KV cache of BPE models. We built a 13.3M-parameter BLT-style model with one cache entry per patch and tried several ways of compressing that cache under a controlled protocol: three seeds, paired bootstrap intervals, random and recency controls, a test split, and tests for every policy. What succeeds: eviction joined to 4-bit quantization keeps 7.4 percent of the cache for +0.0012 BPB; an online SnapKV-style policy beats recency in 24 of 24 comparisons; a cache of fixed size follows documents four times longer than its training better than resetting the context, with constant measured memory and time per byte; the BLT needs 2.8 times fewer FLOPs per byte than a byte-level Transformer. What fails: keeping the patches of high next-byte entropy, the very signal the BLT already computes; it is worse than random in 68 of 72 comparisons, and our error analysis shows that it hoards fragments of one and two bytes. What we cannot assert: whether any of this holds for far larger models or for retrieval tasks, since ours do not retrieve.

### Best result, best evidence

The best practical result is `results/part3/deploy_combo.md` (7.4 percent of the cache bytes, +0.0012 BPB). The best evidence for the negative result is 72 paired comparisons (`results/part3/seeds.md`), the test split (`results/part3/test_split.md`), tenfold training (`results/part3/scale_and_tuning.md`) and the error analysis (`docs/error_analysis_part3.md`). The best sign of care is the list of errors we ourselves discovered and corrected (`CHALLENGES.md`, entries 12, 13 and 16).

### Why this problem, and how it compares with the models the public uses

Why did you choose it? The KV cache is the bottleneck of deployment; tokenizer-free models are alluring but quadruple it; and the BLT already computes an entropy that nobody had put to use on the cache (`docs/proposal_vs_delivered.md`, report section 1).

How does it compare with ChatGPT-class models? Public models such as Llama 2 and Llama 3 read BPE subword tokens (vocabularies of 32k and 128k) and curb the cache in the architecture itself, for instance by grouped-query attention with 8 KV heads in Llama 3 (verified in arXiv:2407.21783, Table 3). We study the same bottleneck on bytes at 13.3M parameters, some 600 times smaller than an 8B model, and therefore compare no quality. What transfers is the protocol, the policies, and the finding that attention-guided eviction serves while the gratuitous entropy signal does not (`report/figures/fig_public.pdf`, `configs/public_models.json`).

Did you do what you proposed? Yes, with stated departures (`docs/proposal_vs_delivered.md`).

### Novelty (3)

What is new? A controlled trial of entropy-ranked cache retention on a byte-level model with patch-granular caches, against random and recency at equal cache bytes, with three seeds per setting, paired bootstrap intervals and a test split (`results/part3/seeds.md`, `results/part3/test_split.md`); an online, tested implementation of H2O and SnapKV-style policies over patch caches (`BLT.forward_online`, `tests/test_blt.py`); and measured streaming results with a bounded cache (`results/part3/stream_16k.md`, `results/part3/pg19_stream.md`). The result of the principal trial is negative.

What is not new? The BLT, StreamingLLM, H2O, SnapKV, KV quantization, and the use of entropy for cache management on subword models are published (`NOVELTY.md`, `report/references.bib`). We do not claim the notion of entropy-guided caching; CONF-KV already uses next-token entropy and already possesses a random-eviction control.

How does your model differ from the BLT? Mean pooling of byte states in place of cross-attention, no n-gram hash embeddings, 13.3M parameters, 3,000 training steps (`docs/block_2_3.md`). It does not beat the BPE Transformer at equal training (`docs/results_summary.md`, gate table).

Why is a negative result worth reporting? Because the hypothesis is cheap and plausible, and others will be tempted by it. We foreclosed the customary escape routes: three seeds, fixed-stride and stride-5 controls, the model's own entropy, four aggregation rules, sink sizes, a byte-level cache, tenfold training, 16,384-byte contexts, a recency-protected variant and a held-out test split (`docs/results_summary.md`, `docs/phase_6_ablations.md`).

How does your work differ from CONF-KV, InfoKV and Scratchpad Patching? CONF-KV uses entropy to decide how much to keep and ranks entries by attention and recency; we rank which patches to keep by entropy alone. InfoKV blends entropy with attention. Scratchpad Patching uses byte-level uncertainty to add slots within patches. We read CONF-KV and the counter-causal paper in full and only the abstracts of InfoKV, Scratchpad Patching and Don't Waste Bits, so for those three our statement of difference is provisional (`NOVELTY.md`, `report/main.tex` section 2).

### Improvement over the previous technique (3)

Does the byte model beat the Part 2 baselines? Against the Kneser-Ney 4-gram, handsomely. Against the BPE Transformer at equal training (3,000 steps), no: 0.7980 against 0.6939 (TinyStories) and 1.6801 against 1.5200 (WikiText). Trained tenfold longer, the WikiText code-mix BLT scores 1.2949 to 1.3118, below the BPE figure, but that comparison grants the BLT ten times the training and a code mix, so it is not like for like (`docs/results_summary.md`, gate table and scale section).

Is entropy patching better than fixed stride? Probably on WikiText (lower in 3 of 3 same-seed pairs, mean gap 0.039 against seed ranges of 0.03 to 0.04) and tied with stride 4 on TinyStories. Three seeds do not establish it (`docs/results_summary.md`, seed section).

Where, then, is the improvement? In cache memory and streaming, not in accuracy: eviction with 4-bit quantization keeps 7.4 percent of the WikiText cache bytes for +0.0012 BPB (`results/part3/deploy_combo.md`); a fixed recency cache beats resetting the context on PG-19 by 0.0086 BPB with 256 patches (`results/part3/pg19_stream.md`); SnapKV-style beats recency in 24 of 24 online comparisons by 0.0002 to 0.0077 BPB (`results/part3/online_policies.md`).

Is the SnapKV-style gain meaningful? It is modest. At 3,000 steps it recovers a fraction of the eviction cost (0.0041 of 0.0098 for seed 0 of TinyStories at 32 patches); at 30,000 steps it lies 0.005 BPB below recency at 64 patches in all three seeds (`results/part3/scale_and_tuning.md`). The intervals exclude zero; the practical gain is slight.

Does the BLT save compute? Against a byte-level Transformer of equal width, yes (11.1 against 30.9 MFLOPs per byte at 1,024 bytes). Against the BPE Transformer, no: 11.1, or 19.6 with the entropy model, against 7.4. The case for the BLT here is cache memory, not FLOPs (`results/part3/flops.md`; analytic, not measured).

Does the patch cache save memory without compression? Not against BPE: at full precision it is about 0.9 times the BPE cache, and a byte-level Transformer needs about four times as much (`results/part3/deploy_memory.md`). The saving comes from eviction and quantization.

### Results and error analysis (2)

Why bits per byte? It is indifferent to the tokenizer, so that the 4-gram, BPE and byte models are scored on identical text. Its limits: it does not measure retrieval, and the Part 2 BPE figure is 0.4 to 0.5 percent flattering (`README.md`).

How do you know the differences are not noise? Every cache comparison is paired by document, with a 95 percent bootstrap interval; the sign holds in every seed. Patching comparisons are not established, because the seed spread (0.03 to 0.14 BPB) is as large as the gaps (`docs/results_summary.md`).

Why does keeping high-entropy patches fail? We do not know for certain. Attention received by old patches correlates negatively with patch entropy in the middle and upper layers of most models (`results/part3/attention_*.md`). In the error analysis the entropy rule keeps almost nothing but the highest-entropy third of patches and forfeits on the lowest-entropy third, which holds 55 to 57 percent of the bytes, and even the highest-entropy third fares worse than under recency (`results/part3/error_analysis.md`, `docs/error_analysis_part3.md`). This is a correlation; no intervention tests it.

Where does eviction hurt most? The increments of per-byte loss are heavy-tailed: the top 1 percent of bytes carry 30.4 percent (TinyStories) and 18.0 percent (WikiText) of the extra loss under entropy against recency, and they are often bytes that the full cache predicts well (`docs/error_analysis_part3.md`).

Was the test split used properly? Once, for 18 WikiText models, with the policies and budgets fixed on validation. Keeping high entropy is worse than random and recency in all 36 cells (`results/part3/test_split.md`). TinyStories has no test split.

Does the result survive more training? It grows: +0.0287 against random at 64 patches at 30,000 steps, against +0.0020 at 3,000 (`results/part3/scale_and_tuning.md`). The TinyStories version overfits and is not used.

Why can a fixed cache beat the full cache? The baseline resets the context every 4,096 bytes; a small recency window carries across the boundary. On PG-19 the whole-book full cache is worse than the reset because positions run 24 times past the training range (`results/part3/stream_16k.md`, `results/part3/pg19_stream.md`).

What can you not conclude? Anything about retrieval (the planted-name test shows the models do not use the earlier mention, `results/part3/deploy_needle.md`); anything about models much larger than 13M parameters; and, on PG-19, whether the entropy signal or long-distance attention is to blame (keeping old patches confounds the two).

### Challenges (2)

What went wrong, and what did you do? The full ledger, with action and outcome, is `CHALLENGES.md`. The ones that altered conclusions: a single-seed patching claim retired after seed variance showed it to be noise; a 16,384-byte conclusion retired when the models proved under-trained; cache byte shares doubled by an accounting error, with every table regenerated; offline attention scores supplanted by an online implementation.

What did you get wrong and retract? The 3.3 percent TinyStories patching gap, the claim that the entropy penalty fades at long context, the claim that the code mix reverses the attention correlation (one outlying seed), and the early cache shares (`docs/results_summary.md`, `CHALLENGES.md`).

How did you cope with the cluster limits? Runs checkpoint, catch SIGUSR1, requeue and resume. The path was verified under a single job id (416139: signalled at step 1,864, resumed, finished at step 24,000). Three earlier attempts failed for avoidable reasons that are on record (`docs/results_summary.md`, requeue section).

### Demonstration and viva (20)

Does the demonstration need a GPU? No. Options 1 to 7 were run on CPU from a fresh clone (`docs/demo_script.md`).

Why does option 6 show almost no difference between policies? A 1.8M-parameter model trained for 300 steps scarcely uses its global cache (BPB 2.147 full against 2.146 to 2.148). The demo says so. Options 4 and 7 use the fully trained models.

In option 7 the entropy policy beat random. Does that contradict the paper? One text of 4,000 bytes is an anecdote; the evidence is 72 paired comparisons with intervals (`results/part3/seeds.md`). In that run recency also beat the full cache, probably because the demo joins 40 unrelated validation stories into one text, whose older members are noise; that explanation is untested (`docs/demo_script.md`).

Why is option 4's cost table so close between policies? On one document of about 1,200 bytes with a budget of 32 patches the differences are 0.01 to 0.02 BPB, the very size they have in the tables (`results/part3/comparison.md`).

How do I regenerate the figures? `python report/make_figures.py` parses `results/part3/*.md`; `python scripts/flops.py` writes `results/part3/flops.md`; `python scripts/error_analysis_part3.py` writes `results/part3/error_analysis.md` (CPU, about two minutes); `python scripts/generation_divergence.py` writes `results/part3/generation_divergence.md` (CPU, about three minutes).
