#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=.
for r in blt_tinystories_entropy blt_tinystories_entropy_s1 blt_tinystories_entropy_s2; do
    [ -f runs/$r/scorer_val_500.json ] || python scripts/eval_scorer.py --run $r --max-docs 500 --fit-docs 300 --budget 32 || exit 1
done
for r in blt_wikitext103_entropy blt_wikitext103_entropy_s1 blt_wikitext103_entropy_s2; do
    [ -f runs/$r/scorer_val_60.json ] || python scripts/eval_scorer.py --run $r --max-docs 60 --fit-docs 20 --budget 64 || exit 1
done
