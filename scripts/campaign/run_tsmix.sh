#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=.
ent=entropy_tinystories_mix
pre=blt_tinystoriesmix
POL="recency random entropy entropy_inv"
[ -f runs/$ent/model.pt ] || python scripts/train_entropy.py configs/entropy/$ent.json || exit 1
python scripts/patch_entropy.py --run $ent || exit 1
python scripts/cache_entropy.py --run $ent --splits train val || exit 1
python scripts/cache_entropy.py --run $ent --dataset code --splits train val || exit 1
for name in ${pre}_entropy ${pre}_entropy_s1 ${pre}_entropy_s2 ${pre}_fixed ${pre}_fixed5; do
    [ -f runs/$name/model.pt ] || python scripts/train_blt.py configs/blt/$name.json || exit 1
    python scripts/eval_blt.py --run $name --max-docs 500 || exit 1
    budgets="32 64"
    case $name in ${pre}_entropy|${pre}_fixed) budgets="12 16 32 64 128";; esac
    python scripts/eval_policies.py --run $name --max-docs 500 --budgets $budgets --policies $POL || exit 1
    python scripts/eval_blt.py --run $name --dataset code --max-docs 200 --out eval_code_200.json || exit 1
    python scripts/eval_policies.py --run $name --dataset code --max-docs 200 --budgets 64 128 --policies $POL --out policies_code_200.json || exit 1
done
run=${pre}_entropy
python scripts/eval_policies.py --run $run --max-docs 500 --budgets 32 64 --policies $POL --signal own --out ablate_own.json || exit 1
for agg in max first sum; do
    python scripts/eval_policies.py --run $run --max-docs 500 --budgets 32 64 --policies $POL --agg $agg --out ablate_agg_$agg.json || exit 1
done
for sink in 0 4; do
    python scripts/eval_policies.py --run $run --max-docs 500 --budgets 32 64 --policies $POL --sink $sink --out ablate_sink$sink.json || exit 1
done
python scripts/eval_quant.py --run $run --max-docs 500 --hi-bits 4 --lo-bits 2 --out quant_hi4_lo2.json || exit 1
python scripts/analyze_attention.py --run $run --max-docs 500 --out results/part3/attention_tinystories_mix.md || exit 1
