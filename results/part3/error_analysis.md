Qualitative and quantitative error analysis of cache eviction, seed-0 entropy-patched BLT models from the `v0.3-part3` release, validation documents, CPU. Regenerate with `python scripts/error_analysis_part3.py`.

## tinystories

100 validation documents, 74,760 bytes (each cut to 4,096 bytes), budget 32 patches, protected set the first patch and the 8 newest. Random is the mean of 3 draws.

| Policy | BPB | Cost over full cache |
|---|---|---|
| full | 0.7553 | +0.0000 |
| recency | 0.7657 | +0.0104 |
| random | 0.7680 | +0.0126 |
| entropy | 0.7794 | +0.0240 |
| entropy_inv | 0.7660 | +0.0107 |

### BPB by entropy tercile of the patch holding the byte (cut points 0.554 and 0.958 nats)

Cost over the full cache, in BPB, within each tercile (the full-cache BPB of the tercile is in the first column).

| Tercile | Share of bytes | Full-cache BPB | recency | random | entropy | entropy_inv |
|---|---|---|---|---|---|---|
| lowest third | 57.4% | 0.4692 | +0.0080 | +0.0091 | +0.0194 | +0.0065 |
| middle third | 31.7% | 0.9468 | +0.0104 | +0.0152 | +0.0272 | +0.0130 |
| highest third | 10.9% | 1.7022 | +0.0225 | +0.0238 | +0.0392 | +0.0253 |

### Which patches the final cache keeps

Share of evictable patches (all but the first and the 8 newest) that are still in the cache at the last patch of the document, by entropy tercile of the patch.

| Policy | lowest third | middle third | highest third |
|---|---|---|---|
| recency | 13.6% | 16.1% | 15.0% |
| random | 13.9% | 15.5% | 15.3% |
| entropy | 0.0% | 0.2% | 44.6% |
| entropy_inv | 44.6% | 0.0% | 0.0% |

Mean length in bytes of the evictable patches that are kept, against all evictable patches (4.63):

| Policy | Mean kept patch length (bytes) | Share of kept patches that are 1 or 2 bytes long |
|---|---|---|
| recency | 4.46 | 31.2% |
| random | 4.49 | 32.9% |
| entropy | 1.15 | 98.0% |
| entropy_inv | 9.78 | 0.0% |

Share of all evictable patches that are 1 or 2 bytes long: 31.4%.

### Which kinds of bytes lose most under entropy eviction

Mean extra loss in bits per byte, by class of the target byte, over the full-cache loss. Positive is worse.

| Byte class | Count | Full-cache loss (bits) | recency | entropy | entropy minus recency |
|---|---|---|---|---|---|
| space or newline | 14,907 | 0.215 | +0.0108 | +0.0127 | +0.0019 |
| digit | 4 | 10.656 | -0.0430 | -0.1260 | -0.0830 |
| uppercase letter | 1,938 | 2.030 | +0.1050 | +0.2476 | +0.1426 |
| word-initial lowercase | 12,987 | 2.221 | +0.0112 | +0.0240 | +0.0128 |
| word-internal lowercase | 42,216 | 0.415 | +0.0069 | +0.0182 | +0.0113 |
| punctuation or other | 2,708 | 1.077 | -0.0095 | +0.0182 | +0.0277 |

Entropy loses to recency on 40.7% of bytes and wins on 33.1%; the largest 1 percent of bytes by loss increase carry 30.4% of the total increase.

### Bytes with the largest loss increase under entropy against recency

| Document | Position | Context and target byte (in brackets) | Full cache (bits) | recency (bits) | entropy (bits) | Patch holding the byte, entropy score (nats) |
|---|---|---|---|---|---|---|
| 7 | 440 | `o, he asked his friends to help him.  Together, [T]` | 1.51 | 1.32 | 8.64 | 0.33 |
| 89 | 791 | `ld be anything she wanted to be. And with that, [L]` | 0.67 | 0.69 | 6.83 | 0.38 |
| 56 | 617 | `. He wanted his truck to be the best. They argue[d]` | 0.02 | 0.01 | 5.97 | 0.44 |
| 29 | 602 | `.   Just as the spider was about to catch Buzz, [B]` | 1.90 | 1.90 | 7.16 | 1.12 |
| 19 | 375 | ` Tom tried to do a trick like the big dog. He ju[m]` | 0.14 | 0.10 | 5.34 | 0.39 |

### Bytes with the largest loss decrease under entropy against recency

| Document | Position | Context and target byte (in brackets) | Full cache (bits) | recency (bits) | entropy (bits) | Patch holding the byte, entropy score (nats) |
|---|---|---|---|---|---|---|
| 0 | 774 | `and Billy looked for big leaves and found some. [R]` | 3.89 | 6.85 | 2.61 | 0.55 |
| 85 | 593 | `ine that would make the spicy feeling go away. B[e]` | 6.68 | 6.95 | 2.77 | 0.91 |
| 56 | 735 | `ucks and see which one is faster!" Johnny agreed[.]` | 4.55 | 4.12 | 0.14 | 0.43 |
| 13 | 644 | `er back to her mom. All the animals cheered for [S]` | 4.04 | 7.35 | 3.46 | 0.50 |
| 44 | 826 | `he little girl never forgot how the robot had to[r]` | 11.38 | 11.45 | 7.87 | 1.10 |

### tinystories: document 0

1201 bytes, 297 patches plus the start patch, 4.0 bytes per patch. Full-cache BPB 0.734.

Text with patch boundaries (`|`), first 420 bytes:

```
Once upon a time|, in a |big |forest, there li|ved a |r|hinoceros |named |R|o|xy. |Ro|x|y |loved to |c|limb. |She |c|limbed |trees, |r|o|cks, and |h|i|lls. |One day, |Ro|x|y |found an |i|cy |h|ill. |She |had |never |seen |anything |like |it| |before. |It was |shiny and |c|old|, |and |she |wanted to |c|limb it.  |Ro|x|y |t|ried to |climb |the |icy |hill, but |it was |very |s|lippery. |She |t|ried |again and |again, |b
```

Patches in the cache at the last patch of the document, budget 32 (first 210 bytes of the kept text, patches in order, `|` between them):

- recency, BPB 0.744 (+0.010 over full): `played |together |all the |time. |And |Ro|x|y |l|earned that |w|ith |a |little |h|elp| |from |a| |f|riend|, |s|he |c|ould |c|l|imb |anything.`
- random, BPB 0.746 (+0.013 over full): `i|lls. |again, |f|alling |h|so |much. |she |Ro|it. |. |icy |x|R|ut |ried to |so |She |From that day on|l|ith |a |s|he |c|ould |c|l|imb |anything.`
- entropy, BPB 0.751 (+0.017 over full): ` |s|f| |B|s|s|Ro| |b|f|b|f|a|s|a|i| |a|a | |a| |s|he |c|ould |c|l|imb |anything.`
- entropy_inv, BPB 0.743 (+0.009 over full): `Once upon a time|, in a |forest, there li|limb. |limbed |One day, |found an |never |seen |ried |again, |anted to |Why are you |ooked |reached the |From that day on|best of friends,|imbed |played |together |And |earned that |riend|s|he `

## wikitext103

30 validation documents, 122,880 bytes (each cut to 4,096 bytes), budget 64 patches, protected set the first patch and the 8 newest. Random is the mean of 3 draws.

| Policy | BPB | Cost over full cache |
|---|---|---|
| full | 1.6575 | +0.0000 |
| recency | 1.6607 | +0.0033 |
| random | 1.6648 | +0.0073 |
| entropy | 1.6730 | +0.0156 |
| entropy_inv | 1.6684 | +0.0109 |

### BPB by entropy tercile of the patch holding the byte (cut points 1.128 and 1.684 nats)

Cost over the full cache, in BPB, within each tercile (the full-cache BPB of the tercile is in the first column).

| Tercile | Share of bytes | Full-cache BPB | recency | random | entropy | entropy_inv |
|---|---|---|---|---|---|---|
| lowest third | 54.7% | 1.2057 | +0.0023 | +0.0054 | +0.0152 | +0.0035 |
| middle third | 31.4% | 1.8943 | +0.0037 | +0.0085 | +0.0146 | +0.0165 |
| highest third | 13.9% | 2.9017 | +0.0062 | +0.0124 | +0.0190 | +0.0274 |

### Which patches the final cache keeps

Share of evictable patches (all but the first and the 8 newest) that are still in the cache at the last patch of the document, by entropy tercile of the patch.

| Policy | lowest third | middle third | highest third |
|---|---|---|---|
| recency | 6.4% | 5.7% | 5.9% |
| random | 5.7% | 6.2% | 6.2% |
| entropy | 0.0% | 0.0% | 18.1% |
| entropy_inv | 18.1% | 0.0% | 0.0% |

Mean length in bytes of the evictable patches that are kept, against all evictable patches (4.45):

| Policy | Mean kept patch length (bytes) | Share of kept patches that are 1 or 2 bytes long |
|---|---|---|
| recency | 4.47 | 34.5% |
| random | 4.38 | 36.1% |
| entropy | 1.13 | 99.7% |
| entropy_inv | 9.20 | 0.0% |

Share of all evictable patches that are 1 or 2 bytes long: 35.0%.

### Which kinds of bytes lose most under entropy eviction

Mean extra loss in bits per byte, by class of the target byte, over the full-cache loss. Positive is worse.

| Byte class | Count | Full-cache loss (bits) | recency | entropy | entropy minus recency |
|---|---|---|---|---|---|
| space or newline | 22,956 | 0.321 | +0.0019 | +0.0038 | +0.0019 |
| digit | 2,444 | 3.436 | +0.0126 | -0.0073 | -0.0199 |
| uppercase letter | 4,627 | 5.276 | +0.0208 | +0.0125 | -0.0083 |
| word-initial lowercase | 14,892 | 3.813 | -0.0005 | +0.0433 | +0.0437 |
| word-internal lowercase | 73,754 | 1.282 | +0.0030 | +0.0155 | +0.0125 |
| punctuation or other | 4,207 | 2.879 | +0.0034 | -0.0002 | -0.0037 |

Entropy loses to recency on 51.0% of bytes and wins on 42.4%; the largest 1 percent of bytes by loss increase carry 18.0% of the total increase.

### Bytes with the largest loss increase under entropy against recency

| Document | Position | Context and target byte (in brackets) | Full cache (bits) | recency (bits) | entropy (bits) | Patch holding the byte, entropy score (nats) |
|---|---|---|---|---|---|---|
| 27 | 1768 | ` , about 16 miles ( 26 km ) to the northeast , i[s]` | 0.84 | 0.86 | 4.15 | 1.43 |
| 11 | 2155 | `nes of US 75 , which are on the bottom level , a[r]` | 2.18 | 1.92 | 5.21 | 1.47 |
| 13 | 1663 | `s dry climate , a few year-round streams provide[ ]` | 1.72 | 1.49 | 4.78 | 1.20 |
| 15 | 1468 | `2006 guide to the churches of Anglesey , suggest[s]` | 0.69 | 0.78 | 3.99 | 0.69 |
| 9 | 3146 | `s " is a blues-style protest song that criticize[s]` | 1.70 | 1.20 | 4.23 | 0.85 |

### Bytes with the largest loss decrease under entropy against recency

| Document | Position | Context and target byte (in brackets) | Full cache (bits) | recency (bits) | entropy (bits) | Patch holding the byte, entropy score (nats) |
|---|---|---|---|---|---|---|
| 7 | 2112 | `touchdown in the AAA B.C. championship game , he[l]` | 7.66 | 8.17 | 3.61 | 0.76 |
| 12 | 2430 | `yn was a field office manager . Noticing that hi[g]` | 9.54 | 9.00 | 6.27 | 0.67 |
| 6 | 3067 | `ical themes and artwork concerns , deeming the a[r]` | 4.83 | 5.65 | 2.94 | 2.39 |
| 23 | 3967 | `rding Highway , a two-lane undivided road , turn[i]` | 5.02 | 4.15 | 1.56 | 0.83 |
| 12 | 2079 | `character in the movie Days of Thunder , played [b]` | 4.55 | 3.82 | 1.23 | 0.98 |

### wikitext103: document 0

4096 bytes, 957 patches plus the start patch, 4.3 bytes per patch. Full-cache BPB 1.719.

Text with patch boundaries (`|`), first 420 bytes:

```
= |H|o|marus |ga|mmaru|s |= |Ho|maru|s |ga|mmaru|s |, |known as |the |E|uropean |lobster |or |common |lobster |, |is |a |s|pecies of |clawed lo|bster |from |the |e|astern |A|tlantic |O|cean |, |Me|diterranean |S|e|a |and |parts of the |Black |S|ea |. |It |is |closely |re|lated to |the |A|merican |lobster |, |H. |a|mericanus |. |It |may |grow |to |a |length of |6|0 |cm ( 2|4 in ) |and |a |mass |of |6| |kilograms ( 1|3
```

Patches in the cache at the last patch of the document, budget 64 (first 210 bytes of the kept text, patches in order, `|` between them):

- recency, BPB 1.725 (+0.006 over full): `. |= = |Distribution = =| |Ho|marus |gammarus |is |found across the| |north-|eastern |A|tlantic |O|cean |from |northern |Norway |to |the |A|zores |and |Mo|r|occo |, |not |including |the |Ba|ltic |S|ea |. |It |is |also |present |in |most |of the |Me|diterranean `
- random, BPB 1.732 (+0.013 over full): `a |s|grow |cm ( 2|mass |of |. |fe , |the |are |ggs |planktonic |lobster |B|gammarus |with |6 |3 |po|, |have |grow |a |young |lo|, |nodules |harp |the |The |associated with |cooking |This |because , |of |ccur |ince |over|. |ricanus |in ) |s|s|and |to |months `
- entropy, BPB 1.752 (+0.033 over full): `o|Ho|E|A|O|S|A|a|6| |a |e|5 |1|e|e|1| |e|e|d |d |a| |d |a|A|a|a|a|a|a|d |a|a|e|a|3|a |a |e|e|1|a|e|1| |a |a|1| | |A|O|A|, |and |along only |the |north-|west |coast of |t`
- entropy_inv, BPB 1.739 (+0.020 over full): `diterranean |merican |coming |ducing |males |fore |= = Description |= = |centimetres ( 2|– |kilograms ( |usually |– |– |veral |– |years |ther |which |generally |associated with |after |leasing |pecies |although |their |nges |pecies |distinguished `

