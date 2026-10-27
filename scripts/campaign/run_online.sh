#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=.
declare -A DOCS=([tinystories]=300 [wikitext103]=60 [tinystoriesmix]=300 [wikitext103mix]=60)
declare -A BUDGETS=([tinystories]="32 64" [wikitext103]="64 128" [tinystoriesmix]="32 64" [wikitext103mix]="64 128")
one() {
    ds=$1
    name=blt_${ds}_$2
    [ -f runs/$name/online.json ] || python scripts/eval_policies.py --run $name --max-docs $3 --budgets $4 --policies recency h2o snap --online --out online.json
}
export -f one
for ds in wikitext103 wikitext103mix tinystories tinystoriesmix; do
    for suf in entropy entropy_s1 entropy_s2; do
        echo "$ds $suf ${DOCS[$ds]} ${BUDGETS[$ds]}"
    done
done | xargs -P 4 -L 1 bash -c 'one "$0" "$1" "$2" "$3 $4"' || exit 1
