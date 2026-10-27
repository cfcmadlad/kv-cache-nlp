#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=.
declare -A DOCS=([tinystories]=500 [wikitext103]=60 [tinystoriesmix]=500 [wikitext103mix]=60)
declare -A BUDGETS=([tinystories]="32 64" [wikitext103]="64 128" [tinystoriesmix]="32 64" [wikitext103mix]="64 128")
[ -f runs/blt_wikitext103mix_entropy/bench.json ] || python scripts/bench_cache.py --run blt_wikitext103mix_entropy --lengths 2048 8192 32768 65536 --budgets 64 128 256 || exit 1
for ds in tinystories wikitext103 tinystoriesmix wikitext103mix; do
    for suf in entropy entropy_s1 entropy_s2; do
        name=blt_${ds}_$suf
        [ -f runs/$name/combo.json ] || python scripts/eval_combo.py --run $name --max-docs ${DOCS[$ds]} --budgets ${BUDGETS[$ds]} --policies recency random entropy snap --bits 16 4 || exit 1
        for b in ${BUDGETS[$ds]}; do
            [ -f runs/$name/hybrid_$b.json ] || python scripts/eval_policies.py --run $name --max-docs ${DOCS[$ds]} --budgets $b --recent $((b / 2)) --policies random entropy entropy_inv --out hybrid_$b.json || exit 1
        done
    done
done
for name in blt_tinystories_entropy blt_tinystories_entropy_s1 blt_wikitext103mix_entropy; do
    [ -f runs/$name/needle.json ] || python scripts/eval_needle.py --run $name --trials 40 || exit 1
done
