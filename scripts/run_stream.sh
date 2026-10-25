#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.
one() {
    name=blt_wikitext103mix_$1
    [ -f runs/$name/stream4k.json ] || python scripts/eval_policies.py --run $name --block 4096 --min-len 8192 --budgets 64 128 256 --policies recency random entropy --out stream4k.json || return 1
    [ -f runs/$name/stream16k.json ] || python scripts/eval_policies.py --run $name --block 16384 --min-len 8192 --budgets 64 128 256 --policies recency random entropy h2o snap --online attn --out stream16k.json
}
export -f one
printf '%s\n' entropy entropy_s1 entropy_s2 | xargs -P 3 -L 1 bash -c 'one "$0"' || exit 1
