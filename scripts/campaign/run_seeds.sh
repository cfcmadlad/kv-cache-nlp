#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=.
declare -A BUDGETS=([tinystories]="32 64" [wikitext103]="64 128")
declare -A DOCS=([tinystories]=500 [wikitext103]=60)
for ds in tinystories wikitext103; do
    for name in blt_${ds}_entropy_s1 blt_${ds}_entropy_s2 blt_${ds}_fixed5; do
        [ -f runs/$name/model.pt ] || python scripts/train_blt.py configs/blt/$name.json || exit 1
        python scripts/eval_blt.py --run $name --max-docs ${DOCS[$ds]} || exit 1
        python scripts/eval_policies.py --run $name --max-docs ${DOCS[$ds]} --budgets ${BUDGETS[$ds]} \
            --policies recency random entropy entropy_inv || exit 1
    done
done
