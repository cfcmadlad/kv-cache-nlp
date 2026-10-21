Models evaluated on the first 200 validation files of the code set (Python, Java, JavaScript; bigcode/the-stack-smol). Rows without mix were trained on prose only (out of domain, including their entropy model); wikitext103mix was trained with 15% code. Seed-0 models, 95% document bootstrap intervals (2000 resamples).

| Trained on | Full-cache BPB [95% CI] | Mean patch length | Budget | recency | random | entropy | entropy_inv | entropy - random | entropy - recency |
|---|---|---|---|---|---|---|---|---|---|
| tinystories | 7.6384 [7.4403, 7.8244] | 1.56 | 64 | 7.6573 | 7.6695 | 7.8061 | 7.6232 | +0.1365 [+0.1241, +0.1484] | +0.1487 [+0.1360, +0.1612] |
| tinystories | 7.6384 [7.4403, 7.8244] | 1.56 | 128 | 7.6335 | 7.6550 | 7.8375 | 7.6104 | +0.1822 [+0.1651, +0.1981] | +0.2038 [+0.1864, +0.2194] |
| wikitext103 | 7.1680 [6.9632, 7.3584] | 2.514 | 64 | 7.2437 | 7.2751 | 7.4149 | 7.2744 | +0.1394 [+0.1265, +0.1506] | +0.1707 [+0.1547, +0.1844] |
| wikitext103 | 7.1680 [6.9632, 7.3584] | 2.514 | 128 | 7.2046 | 7.2384 | 7.4020 | 7.2321 | +0.1632 [+0.1471, +0.1766] | +0.1969 [+0.1787, +0.2122] |
| tinystoriesmix | 1.8842 [1.8254, 1.9487] | 2.37 | 64 | 1.8871 | 1.8888 | 1.8991 | 1.8927 | +0.0103 [+0.0090, +0.0117] | +0.0120 [+0.0108, +0.0133] |
| tinystoriesmix | 1.8842 [1.8254, 1.9487] | 2.37 | 128 | 1.8854 | 1.8869 | 1.8947 | 1.8892 | +0.0077 [+0.0067, +0.0089] | +0.0092 [+0.0082, +0.0102] |
| wikitext103mix | 1.7476 [1.6975, 1.7977] | 4.0 | 64 | 1.7496 | 1.7501 | 1.7537 | 1.7515 | +0.0036 [+0.0030, +0.0044] | +0.0041 [+0.0034, +0.0051] |
| wikitext103mix | 1.7476 [1.6975, 1.7977] | 4.0 | 128 | 1.7485 | 1.7489 | 1.7510 | 1.7496 | +0.0020 [+0.0016, +0.0025] | +0.0025 [+0.0019, +0.0032] |