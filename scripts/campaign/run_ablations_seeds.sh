#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=.
declare -A DOCS=([tinystories]=500 [wikitext103]=60)
declare -A BUDGETS=([tinystories]="32 64" [wikitext103]="64 128")
POL="recency random entropy entropy_inv"
for suf in entropy_s1 entropy_s2; do
    for ds in tinystories wikitext103; do
        run=blt_${ds}_$suf
        n=${DOCS[$ds]}
        b=${BUDGETS[$ds]}
        [ -f runs/$run/ablate_own.json ] || python scripts/eval_policies.py --run $run --max-docs $n --budgets $b --policies $POL --signal own --out ablate_own.json || exit 1
        for agg in max first sum; do
            [ -f runs/$run/ablate_agg_$agg.json ] || python scripts/eval_policies.py --run $run --max-docs $n --budgets $b --policies $POL --agg $agg --out ablate_agg_$agg.json || exit 1
        done
        for sink in 0 4; do
            [ -f runs/$run/ablate_sink$sink.json ] || python scripts/eval_policies.py --run $run --max-docs $n --budgets $b --policies $POL --sink $sink --out ablate_sink$sink.json || exit 1
        done
    done
done
for name in blt_tinystories_fixed1_s1 blt_tinystories_fixed1_s2; do
    [ -f runs/$name/model.pt ] || continue
    [ -f runs/$name/policies_val_500.json ] || python scripts/eval_policies.py --run $name --max-docs 500 --budgets 32 64 144 288 --policies $POL || exit 1
done
