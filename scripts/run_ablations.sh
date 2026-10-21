#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.
declare -A DOCS=([tinystories]=500 [wikitext103]=60)
declare -A BUDGETS=([tinystories]="32 64" [wikitext103]="64 128")
POL="recency random entropy entropy_inv"
for ds in tinystories wikitext103; do
    run=blt_${ds}_entropy
    n=${DOCS[$ds]}
    b=${BUDGETS[$ds]}
    python scripts/eval_policies.py --run $run --max-docs $n --budgets $b --policies $POL --signal own --out ablate_own.json || exit 1
    for agg in max first sum; do
        python scripts/eval_policies.py --run $run --max-docs $n --budgets $b --policies $POL --agg $agg --out ablate_agg_$agg.json || exit 1
    done
    for sink in 0 4; do
        python scripts/eval_policies.py --run $run --max-docs $n --budgets $b --policies $POL --sink $sink --out ablate_sink$sink.json || exit 1
    done
done
name=blt_tinystories_fixed1
[ -f runs/$name/model.pt ] || python scripts/train_blt.py configs/$name.json || exit 1
python scripts/eval_blt.py --run $name --max-docs 500 || exit 1
python scripts/eval_policies.py --run $name --max-docs 500 --budgets 32 64 144 288 --policies $POL || exit 1
