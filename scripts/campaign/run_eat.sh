#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=.
ds=${1:-tinystories}
declare -A DOCS=([tinystories]=500 [wikitext103]=60)
declare -A BUDGETS=([tinystories]="32 64" [wikitext103]="64 128")
for suf in "" _s1 _s2; do
    for reg in recency random entropy mixed; do
        name=blt_${ds}_eat_${reg}${suf}
        [ -f runs/$name/model.pt ] || python scripts/train_blt.py configs/blt/$name.json || exit 1
        [ -f runs/$name/policies_val_${DOCS[$ds]}.json ] || python scripts/eval_policies.py --run $name --max-docs ${DOCS[$ds]} --budgets ${BUDGETS[$ds]} --policies recency random entropy snap || exit 1
    done
done
