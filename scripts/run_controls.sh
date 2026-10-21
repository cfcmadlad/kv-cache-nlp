#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.
declare -A DOCS=([tinystories]=500 [wikitext103]=60 [tinystoriesmix]=500 [wikitext103mix]=60)
declare -A BUDGETS=([tinystories]="32 64" [wikitext103]="64 128" [tinystoriesmix]="32 64" [wikitext103mix]="64 128")
POL="recency random entropy entropy_inv"
for ds in tinystories wikitext103 tinystoriesmix wikitext103mix; do
    for kind in fixed fixed5; do
        for s in 1 2; do
            name=blt_${ds}_${kind}_s$s
            [ -f runs/$name/model.pt ] || python scripts/train_blt.py configs/$name.json || exit 1
            python scripts/eval_blt.py --run $name --max-docs ${DOCS[$ds]} || exit 1
            python scripts/eval_policies.py --run $name --max-docs ${DOCS[$ds]} --budgets ${BUDGETS[$ds]} --policies $POL || exit 1
        done
    done
done
for s in 1 2; do
    name=blt_tinystories_fixed1_s$s
    [ -f runs/$name/model.pt ] || python scripts/train_blt.py configs/$name.json || exit 1
    python scripts/eval_blt.py --run $name --max-docs 500 || exit 1
    python scripts/eval_policies.py --run $name --max-docs 500 --budgets 32 64 144 288 --policies $POL || exit 1
done
for ds in tinystories wikitext103 wikitext103mix; do
    for s in s1 s2; do
        name=blt_${ds}_entropy_$s
        python scripts/eval_quant.py --run $name --max-docs ${DOCS[$ds]} --hi-bits 4 --lo-bits 2 --out quant_hi4_lo2.json || exit 1
        python scripts/analyze_attention.py --run $name --max-docs ${DOCS[$ds]} --out results/part3/attention_${name}.md || exit 1
    done
done
