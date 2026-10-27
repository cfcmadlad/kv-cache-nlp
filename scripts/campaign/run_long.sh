#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=.
for kind in entropy fixed; do
    name=blt_wikitext103mix_L16k_$kind
    [ -f runs/$name/model.pt ] || python scripts/train_blt.py configs/blt/$name.json || exit 1
    python scripts/eval_blt.py --run $name --max-docs 60 || exit 1
    python scripts/eval_policies.py --run $name --max-docs 60 --budgets 64 128 256 512 --policies recency random entropy entropy_inv h2o snap || exit 1
done
