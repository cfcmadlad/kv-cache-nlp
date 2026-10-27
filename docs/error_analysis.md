# Anatomy of an eviction: what a bounded cache does to the text

The Part 1 proposal pledged a qualitative scrutiny of where eviction wounds the prediction, and this document redeems the pledge. Everything below is drawn from `results/part3/error_analysis.md` and `results/part3/generation_divergence.md`, written by `python scripts/error_analysis_part3.py` (CPU, about two minutes) and `python scripts/generation_divergence.py` (CPU, about three) from the seed-0 entropy-patched models of the `v0.3-part3` release and the validation documents. The arrangement: TinyStories, 100 documents (74,760 bytes) at a budget of 32 patches; WikiText-103, 30 documents (122,880 bytes, each cut to 4,096 bytes) at 64 patches; the first patch and the eight newest always spared; random is the mean of three draws. Examples are transcribed from those files, unaltered. Whatever is offered in explanation is marked a hypothesis.

## Where the extra loss falls

BPB over these documents, and the cost of each policy over the full cache:

| Set | full | recency | random | keep high entropy | keep low entropy |
|---|---|---|---|---|---|
| TinyStories, 32 patches | 0.7553 | +0.0104 | +0.0126 | +0.0240 | +0.0107 |
| WikiText, 64 patches | 1.6575 | +0.0033 | +0.0073 | +0.0156 | +0.0109 |

Split by the entropy tercile of the patch that holds the byte (cost over the full cache, BPB):

| Set | Tercile (share of bytes) | Full-cache BPB | recency | keep high entropy | keep low entropy |
|---|---|---|---|---|---|
| TinyStories | lowest (57.4%) | 0.4692 | +0.0080 | +0.0194 | +0.0065 |
| TinyStories | middle (31.7%) | 0.9468 | +0.0104 | +0.0272 | +0.0130 |
| TinyStories | highest (10.9%) | 1.7022 | +0.0225 | +0.0392 | +0.0253 |
| WikiText | lowest (54.7%) | 1.2057 | +0.0023 | +0.0152 | +0.0035 |
| WikiText | middle (31.4%) | 1.8943 | +0.0037 | +0.0146 | +0.0165 |
| WikiText | highest (13.9%) | 2.9017 | +0.0062 | +0.0190 | +0.0274 |

- Every policy loses most on the high-entropy third, which is where the full-cache loss is also highest (1.70 and 2.90 BPB).
- Keeping high-entropy patches does not protect the high-entropy bytes. Their extra loss is +0.0392 (TinyStories) and +0.0190 (WikiText) under the entropy rule against +0.0225 and +0.0062 under recency. It also costs the easy bytes: on the lowest third, +0.0194 and +0.0152 against +0.0080 and +0.0023.
- Keeping low-entropy patches is the opposite on TinyStories (close to recency in every tercile) and is worst on the high third on WikiText (+0.0274), where it matches the earlier finding that low-entropy retention depends on the data (`docs/results_summary.md`).

## What each rule keeps

Share of evictable patches (all but the first and 8 newest) still in the cache at the last patch of a document, by entropy tercile:

| Set | Rule | lowest third | middle third | highest third |
|---|---|---|---|---|
| TinyStories | recency | 13.6% | 16.1% | 15.0% |
| TinyStories | keep high entropy | 0.0% | 0.2% | 44.6% |
| WikiText | recency | 6.4% | 5.7% | 5.9% |
| WikiText | keep high entropy | 0.0% | 0.0% | 18.1% |

Patch length of what is kept (all evictable patches average 4.63 bytes on TinyStories and 4.45 on WikiText):

| Set | Rule | Mean kept patch length (bytes) | Kept patches that are 1 or 2 bytes long |
|---|---|---|---|
| TinyStories | recency | 4.46 | 31.2% |
| TinyStories | keep high entropy | 1.15 | 98.0% |
| TinyStories | keep low entropy | 9.78 | 0.0% |
| WikiText | recency | 4.47 | 34.5% |
| WikiText | keep high entropy | 1.13 | 99.7% |
| WikiText | keep low entropy | 9.20 | 0.0% |

The mean-entropy rule retains almost nothing but patches of one or two bytes. This is a confound of length: a patch begins where the next byte is surprising, so the surprising byte frequently ends in a patch of its own, and the mean over so short a patch runs high. It accounts for part of what the entropy cache looks like, though not for the whole result: the sum rule, which favours long patches, ties random on TinyStories and is worse on WikiText, and the maximum rule is closer to random but still worse in most seeds (`docs/results_summary.md`, ablation sections), so no scoring rule we tried is robustly better than random.

## Which bytes pay

Extra loss over the full cache in bits per byte, by class of the target byte:

| Set | Class | Count | recency | keep high entropy | difference |
|---|---|---|---|---|---|
| TinyStories | uppercase letter | 1,938 | +0.1050 | +0.2476 | +0.1426 |
| TinyStories | word-initial lowercase | 12,987 | +0.0112 | +0.0240 | +0.0128 |
| TinyStories | word-internal lowercase | 42,216 | +0.0069 | +0.0182 | +0.0113 |
| WikiText | word-initial lowercase | 14,892 | -0.0005 | +0.0433 | +0.0437 |
| WikiText | word-internal lowercase | 73,754 | +0.0030 | +0.0155 | +0.0125 |
| WikiText | uppercase letter | 4,627 | +0.0208 | +0.0125 | -0.0083 |
| WikiText | digit | 2,444 | +0.0126 | -0.0073 | -0.0199 |

The entropy rule helps a little on digits and capitals in WikiText (bytes that announce rare and particular tokens) and loses on word beginnings and word interiors. The harm is concentrated: entropy is worse than recency on 40.7% (TinyStories) and 51.0% (WikiText) of bytes, better on 33.1% and 42.4%, and the largest 1 percent of bytes by loss increase carry 30.4% and 18.0% of the total increase.

## Eight examples (real output)

**1. Patches on a TinyStories text** (document 0, 1,201 bytes, 297 patches, 4.0 bytes per patch, full-cache BPB 0.734; `|` marks a boundary):

```
Once upon a time|, in a |big |forest, there li|ved a |r|hinoceros |named |R|o|xy. |Ro|x|y |loved to |c|limb. |She |c|limbed |trees, |r|o|cks, and |h|i|lls. |One day, |Ro|x|y |found an |i|cy |h|ill. |She |had |never |seen |anything |like |it| |before. |It was |shiny and |c|old|, |and |she |wanted to |c|limb it.  |Ro|x|y |t|ried to |climb |the |icy |hill, but |it was |very |s|lippery. |She |t|ried |again and |again, |b
```

The boundaries dissect the name Roxy into `R|o|xy` and `Ro|x|y` and fall before surprising letters (`c|limb`, `r|o|cks`), while predictable function words (`She `, `never `, `anything `) remain whole patches.

**2. What each rule keeps at the end of document 0** (budget 32; first part of the kept text):
- recency, BPB 0.744 (+0.010 over full): `played |together |all the |time. |And |Ro|x|y |l|earned that |w|ith |a |little |h|elp| |from |a| |f|riend|, |s|he |c|ould |c|l|imb |anything.`
- random, BPB 0.746 (+0.013): `i|lls. |again, |f|alling |h|so |much. |she |Ro|it. |. |icy |x|R|ut |ried to |so |She |From that day on|l|ith |a |s|he |c|ould |c|l|imb |anything.`
- keep high entropy, BPB 0.751 (+0.017): ` |s|f| |B|s|s|Ro| |b|f|b|f|a|s|a|i| |a|a | |a| |s|he |c|ould |c|l|imb |anything.`
- keep low entropy, BPB 0.743 (+0.009): `Once upon a time|, in a |forest, there li|limb. |limbed |One day, |found an |never |seen |ried |again, |anted to |Why are you |ooked |reached the |From that day on|best of friends,|imbed |played |together |And |earned that |riend|s|he `

The high-entropy cache is a rosary of one-letter fragments (`s|f| |B|s|s`); it keeps the fragment `Ro` of the name but not `Roxy`. The low-entropy cache holds whole words and phrases from the opening of the story.

**3. Patches and a high-entropy cache on WikiText** (document 0, 4,096 bytes, 957 patches, 4.3 bytes per patch, full-cache BPB 1.719):

```
= |H|o|marus |ga|mmaru|s |= |Ho|maru|s |ga|mmaru|s |, |known as |the |E|uropean |lobster |or |common |lobster |, |is |a |s|pecies of |clawed lo|bster |from |the |e|astern |A|tlantic |O|cean |, |Me|diterranean |S|e|a |and |parts of the |Black |S|ea |. |It |is |closely |re|lated to |the |A|merican |lobster |, |H. |a|mericanus |. |It |may |grow |to |a |length of |6|0 |cm ( 2|4 in ) |and |a |mass |of |6| |kilograms ( 1|3
```

At 64 patches, BPB is 1.725 under recency (+0.006 over full), 1.732 under random (+0.013), 1.739 under keep low entropy (+0.020) and 1.752 under keep high entropy (+0.033). The high-entropy cache at the last patch is `o|Ho|E|A|O|S|A|a|6| |a |e|5 |1|e|e|1| |e|e|d |d |a| |d |a|A|a|a|a|a|a|a|d |a|a|e|a|3|a |a |e|e|1|a|e|1| |a |a|1| | |A|O|A|, |and |along only |the |north-|west |coast of |t`: capitals, digits and vowels, with not a word among them. The recency cache holds the section on distribution (`Norway |to |the |A|zores |and |Mo|r|occo`).

**4. Largest loss increase, TinyStories** (document 7, byte 440, target in brackets): `o, he asked his friends to help him.  Together, [T]`. Full cache 1.51 bits, recency 1.32, keep high entropy 8.64. A capital opening the next word after a comma costs seven extra bits when the cache holds fragments.

**5. Largest loss increase, locally predictable byte** (document 56, byte 617): `. He wanted his truck to be the best. They argue[d]`. Full cache 0.02 bits, recency 0.01, keep high entropy 5.97. The byte is almost certain from the last few bytes, yet it costs six bits under entropy eviction. Hypothesis (untested): the global state that feeds the decoder is computed from the attended cache, and a cache of fragments yields a distorted state even for easy bytes. The eight newest patches are protected, so the recent bytes are present; what changes is everything older.

**6. Largest loss increase, WikiText** (document 27, byte 1768): `, about 16 miles ( 26 km ) to the northeast , i[s]`. Full cache 0.84 bits, recency 0.86, keep high entropy 4.15. The same pattern in a different corpus: an easy continuation rendered expensive.

**7. Where entropy eviction helps** (TinyStories document 13, byte 644): `er back to her mom. All the animals cheered for [S]`. Full cache 4.04 bits, recency 7.35, keep high entropy 3.46. The target is the first letter of the name Sam, which appears several times earlier in the story (checked in the text). Recency scores 7.35 bits, worse than the full cache; the entropy rule scores 3.46, better than the full cache itself. We did not verify whether the entropy cache held a patch containing the name. A WikiText case (document 7, byte 2112): `touchdown in the AAA B.C. championship game , he[l]`: 7.66, 8.17 and 3.61 bits. In both the entropy rule beats recency on a surprising byte. Entropy eviction also beats the full cache on some bytes (for example TinyStories document 56, byte 735, `Johnny agreed[.]`: full 4.55 bits, entropy 0.14), so the full cache is not invariably the best prediction; the attention over many entries is noisy in these small models.

**8. Generations that part company** (`results/part3/generation_divergence.md`; greedy continuation of the same prompt, eighty bytes, entropy patches and eviction masks recomputed at every step). TinyStories document 0, prompt of 700 bytes, 32 patches: the full cache continues `find your friends."  Roxy was so happy to help Roxy.`; recency diverges at byte 17 (`find your friends to climb the leaves."`), random and keep-low-entropy at byte 10 (`find your home and they would...`), and keep-high-entropy at byte 25 (`Roxyy said, "Thank you, Billy.`). WikiText document 1, prompt of 1,500 bytes, 64 patches: recency stays with the full cache until byte 65, whereas random and keep-high-entropy diverge at the very first byte (`r the 1970 [dash] 1996 season`). Greedy text from models so small is repetitive (`the state of the state`), and on these four prompts no policy reproduces the true continuation better than the full cache by more than three bytes of eighty; the examples show where a compressed cache alters the prediction, not how good the prose is.

## Reading

- The entropy rule does keep a different set of patches: almost exclusively high-entropy, one-byte fragments. Now and then that helps (example 7); on average it hurts, because such fragments carry little context worth reusing and because the rule discards the low-entropy patches that carry words and phrases.
- The damage is concentrated in a small share of bytes (the top 1 percent carry 18.0% to 30.4% of the extra loss), and many of those bytes are easy for the full cache.
- This agrees with the attention result, in which old high-entropy patches receive less attention than low-entropy ones in the middle and upper layers of most models (`results/part3/attention_*.md`). It is a correlation; no intervention here separates an effect of length, an effect of entropy and a distortion of the global state.
- One seed per dataset and a fixed budget: the tercile and byte-class tables are descriptive, not tests. The tests are the paired comparisons in `results/part3/seeds.md`.
## Mitigation

The pattern is that keep-high-entropy hoards short fragments of little context and discards the word-sized patches the decoder reads. Three remedies were tested. Shielding a recency window and spending only half the slots on entropy shrinks the penalty but never removes it (worse than recency in all 3 seeds at both budgets on TinyStories, `results/part3/deploy_hybrid.md`). Ranking old patches by the attention they receive (SnapKV-style) beats recency in 24 of 24 comparisons (`results/part3/online_policies.md`). Quantizing every entry to 4 bits instead of evicting costs at most 0.0014 BPB. A floor on patch length, so that fragments of one or two bytes cannot be kept, follows from the diagnosis but was not tested.

