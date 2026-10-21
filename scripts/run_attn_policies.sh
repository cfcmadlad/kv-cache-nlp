#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.
declare -A DOCS=([tinystories]=500 [wikitext103]=60 [tinystoriesmix]=500 [wikitext103mix]=60)
declare -A BUDGETS=([tinystories]="32 64" [wikitext103]="64 128" [tinystoriesmix]="32 64" [wikitext103mix]="64 128")
for ds in tinystories wikitext103 tinystoriesmix wikitext103mix; do
    for suf in entropy entropy_s1 entropy_s2 fixed fixed5 fixed_s1 fixed_s2 fixed5_s1 fixed5_s2; do
        name=blt_${ds}_$suf
        [ -f runs/$name/model.pt ] || continue
        python scripts/eval_policies.py --run $name --max-docs ${DOCS[$ds]} --budgets ${BUDGETS[$ds]} --policies h2o snap --out policies_attn.json || exit 1
    done
done
for name in blt_tinystories_fixed1 blt_tinystories_fixed1_s1 blt_tinystories_fixed1_s2; do
    [ -f runs/$name/model.pt ] || continue
    python scripts/eval_policies.py --run $name --max-docs 500 --budgets 32 64 144 288 --policies h2o snap --out policies_attn.json || exit 1
done
