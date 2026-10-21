#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.
for ds in tinystories wikitext103; do
    ent=entropy_$ds
    run=blt_${ds}_entropy
    python scripts/cache_entropy.py --run $ent --dataset code --splits val || exit 1
    python scripts/eval_blt.py --run $run --dataset code --max-docs 200 --out eval_code_200.json || exit 1
    python scripts/eval_policies.py --run $run --dataset code --max-docs 200 --budgets 64 128 \
        --policies recency random entropy entropy_inv --out policies_code_200.json || exit 1
done
