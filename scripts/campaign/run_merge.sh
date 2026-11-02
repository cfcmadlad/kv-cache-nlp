#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=.
for r in blt_tinystories_entropy blt_tinystories_entropy_s1 blt_tinystories_entropy_s2; do
    [ -f runs/$r/merge_val_500.json ] || python scripts/eval_merge.py --run $r --max-docs 500 --budget 32 || exit 1
done
for r in blt_wikitext103_entropy blt_wikitext103_entropy_s1 blt_wikitext103_entropy_s2; do
    [ -f runs/$r/merge_val_60.json ] || python scripts/eval_merge.py --run $r --max-docs 60 --budget 64 || exit 1
done
