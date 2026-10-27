#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=.
train() {
    [ -f runs/$1/model.pt ] || python scripts/train_blt.py configs/blt/$1.json
}
evaluate() {
    name=$1
    n=$2
    [ -f runs/$name/eval_val_$n.json ] || python scripts/eval_blt.py --run $name --max-docs $n || return 1
    [ -f runs/$name/policies_val_$n.json ] || python scripts/eval_policies.py --run $name --max-docs $n --budgets $3 --policies recency random entropy entropy_inv h2o snap
}
export -f train evaluate
printf '%s\n' blt_wikitext103mix_L16k_fixed_b8 blt_wikitext103mix_entropy_x10_s1 blt_wikitext103mix_entropy_x10_s2 | xargs -P 3 -L 1 bash -c 'train "$0"' || exit 1
{
    echo "blt_wikitext103mix_L16k_entropy_b8 60 64 128 256 512"
    echo "blt_wikitext103mix_L16k_fixed_b8 60 64 128 256 512"
    echo "blt_wikitext103mix_entropy_x10_s1 60 64 128"
    echo "blt_wikitext103mix_entropy_x10_s2 60 64 128"
} | xargs -P 4 -L 1 bash -c 'evaluate "$0" "$1" "${*:2}"' || exit 1
