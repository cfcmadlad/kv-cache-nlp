# BPB comparison

### tinystories

| model | train_docs | val_docs | train_time_sec | train_bpb_sample | val_bpb | order | discounts | num_params | n_layer | n_embd | block_size |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kneser_ney_ngram | 25000 | 2000 | 39.4 | 1.8775 | 1.8477 | 4 | 0.7059, 0.5935, 0.5528, 0.5451 | — | — | — | — |
| bpe_transformer | 30000 | 500 | 780.9 | 0.7745 | 0.6939 | — | — | 13817856 | 6 | 384 | 256 |

### wikitext103

| model | train_docs | val_docs | train_time_sec | train_bpb_sample | val_bpb | order | discounts | num_params | n_layer | n_embd | block_size |
|---|---|---|---|---|---|---|---|---|---|---|---|
| kneser_ney_ngram | 4500 | 60 | 149.5 | 2.3993 | 2.4103 | 4 | 0.3617, 0.5699, 0.6561, 0.5618 | — | — | — | — |
| bpe_transformer | 29002 | 60 | 586.0 | 1.5459 | 1.52 | — | — | 13817856 | 6 | 384 | 256 |
