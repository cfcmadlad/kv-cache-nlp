Matched-document BPB (lower is better). Test splits are not used.

### tinystories val, first 500 docs

| Model | BPB | Bytes per patch or token | Global KV bytes per 1000 input bytes |
|---|---|---|---|
| Kneser-Ney byte 4-gram | 1.8085 | n/a | n/a |
| BPE transformer, reported | 0.6939 | 4.145 | 2,223,451 |
| BPE transformer, corrected | 0.6978 | 4.145 | 2,223,451 |
| Entropy model (3.3M) | 0.8438 | 4.5 | n/a |
| BLT fixed patching (13.3M) | 0.7726 | 3.992 | 2,308,577 |
| BLT entropy patching (13.3M) | 0.7980 | 4.499 | 2,048,288 |

### wikitext103 val, first 60 docs

| Model | BPB | Bytes per patch or token | Global KV bytes per 1000 input bytes |
|---|---|---|---|
| Kneser-Ney byte 4-gram | 2.4103 | n/a | n/a |
| BPE transformer, reported | 1.5200 | 3.983 | 2,314,037 |
| BPE transformer, corrected | 1.5257 | 3.983 | 2,314,037 |
| Entropy model (3.3M) | 1.6983 | 4.5 | n/a |
| BLT fixed patching (13.3M) | 1.7600 | 4 | 2,304,192 |
| BLT entropy patching (13.3M) | 1.6801 | 4.498 | 2,048,914 |

Fixed overhead not in the table: BLT local encoder and decoder windows 0.79 MB, entropy model window 2.10 MB (fp16).