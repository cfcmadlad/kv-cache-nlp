#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=.
[ -f runs/blt_tinystories_entropy/window_val_200.json ] || python scripts/eval_window.py --run blt_tinystories_entropy --max-docs 200 || exit 1
[ -f runs/blt_wikitext103_entropy/window_val_60.json ] || python scripts/eval_window.py --run blt_wikitext103_entropy --max-docs 60 || exit 1
[ -f data/bytes/pg19/entropy_entropy_wikitext103_mix/test.npy ] || python scripts/cache_entropy.py --run entropy_wikitext103_mix --dataset pg19 --splits test || exit 1
r=blt_wikitext103mix_entropy_x10
[ -f runs/$r/stream_pg19_0_5_renum.json ] || python scripts/eval_stream.py --run $r --books 0 5 --renumber --out stream_pg19_0_5_renum.json || exit 1
