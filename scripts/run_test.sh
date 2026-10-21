#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.
POL="recency random entropy entropy_inv h2o snap"
python scripts/cache_entropy.py --run entropy_wikitext103 --splits test || exit 1
python scripts/cache_entropy.py --run entropy_wikitext103_mix --splits test || exit 1
for ds in wikitext103 wikitext103mix; do
    for suf in entropy entropy_s1 entropy_s2 fixed fixed_s1 fixed_s2 fixed5 fixed5_s1 fixed5_s2; do
        name=blt_${ds}_$suf
        [ -f runs/$name/policies_test_62.json ] && continue
        python scripts/eval_blt.py --run $name --split test || exit 1
        python scripts/eval_policies.py --run $name --split test --budgets 64 128 --policies $POL || exit 1
    done
done
