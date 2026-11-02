#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=.
for suf in "" _s1 _s2; do
    for reg in recency random entropy mixed; do
        name=blt_wikitext103_eat_${reg}${suf}
        [ -f runs/$name/policies_test_62.json ] || python scripts/eval_policies.py --run $name --split test --max-docs 62 --budgets 64 128 --policies recency random entropy snap || exit 1
    done
done
