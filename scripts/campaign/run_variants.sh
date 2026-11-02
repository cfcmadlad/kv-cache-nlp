#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=.
for r in blt_tinystories_entropy blt_tinystories_entropy_s1 blt_tinystories_entropy_s2; do
    [ -f runs/$r/variants_val_500.json ] || python scripts/eval_variants.py --run $r --max-docs 500 --budgets 32 --extra-recency 12 16 20 24 || exit 1
done
for r in blt_wikitext103_entropy blt_wikitext103_entropy_s1 blt_wikitext103_entropy_s2; do
    [ -f runs/$r/variants_val_60.json ] || python scripts/eval_variants.py --run $r --max-docs 60 --budgets 64 --extra-recency 20 32 40 48 || exit 1
done
