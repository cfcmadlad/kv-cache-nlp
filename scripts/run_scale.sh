#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.
declare -A DOCS=([tinystories]=500 [wikitext103mix]=60)
declare -A BUDGETS=([tinystories]="32 64" [wikitext103mix]="64 128")
train() {
    [ -f runs/$1/model.pt ] || python scripts/train_blt.py configs/$1.json
}
evaluate() {
    name=$1
    n=$2
    [ -f runs/$name/eval_val_$n.json ] || python scripts/eval_blt.py --run $name --max-docs $n || return 1
    [ -f runs/$name/policies_val_$n.json ] || python scripts/eval_policies.py --run $name --max-docs $n --budgets $3 --policies recency random entropy entropy_inv h2o snap
}
export -f train evaluate
printf '%s\n' blt_wikitext103mix_L16k_entropy_b8 blt_wikitext103mix_L16k_entropy_lr3 blt_wikitext103mix_L16k_entropy_b8lr3 \
    blt_tinystories_entropy_x10 blt_tinystories_fixed_x10 blt_wikitext103mix_entropy_x10 blt_wikitext103mix_fixed_x10 | xargs -P 1 -L 1 bash -c 'train "$0"' || exit 1
for name in blt_wikitext103mix_L16k_entropy_b8 blt_wikitext103mix_L16k_entropy_lr3 blt_wikitext103mix_L16k_entropy_b8lr3; do
    [ -f runs/$name/eval_val_60.json ] || python scripts/eval_blt.py --run $name --max-docs 60 || exit 1
done
for ds in tinystories wikitext103mix; do
    for kind in entropy fixed; do
        echo "blt_${ds}_${kind}_x10 ${DOCS[$ds]} ${BUDGETS[$ds]}"
    done
done | xargs -P 4 -L 1 bash -c 'evaluate "$0" "$1" "$2 $3"' || exit 1
for ds in tinystories wikitext103mix; do
    [ -f runs/blt_${ds}_entropy_x10/needle.json ] || python scripts/eval_needle.py --run blt_${ds}_entropy_x10 --trials 40 || exit 1
done
