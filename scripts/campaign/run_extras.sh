#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=.
for r in blt_tinystories_entropy blt_tinystories_entropy_s1 blt_tinystories_entropy_s2; do
    [ -f runs/$r/extras_val_500.json ] || python scripts/eval_extras.py --run $r --max-docs 500 --budget 32 || exit 1
done
for r in blt_wikitext103_entropy blt_wikitext103_entropy_s1 blt_wikitext103_entropy_s2; do
    [ -f runs/$r/extras_val_60.json ] || python scripts/eval_extras.py --run $r --max-docs 60 --budget 64 || exit 1
done
for r in blt_tinystories_entropy blt_wikitext103_entropy; do
    d=500; [ $r = blt_wikitext103_entropy ] && d=60
    [ -f runs/$r/frontier.json ] || python scripts/eval_combo.py --run $r --max-docs $d --policies recency snap --budgets 16 24 32 48 64 96 128 192 256 --bits 16 8 4 3 2 --out frontier.json || exit 1
done
