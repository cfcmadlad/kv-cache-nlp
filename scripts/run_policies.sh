#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH=.
POL="recency random entropy entropy_inv"
python scripts/eval_policies.py --run blt_tinystories_entropy --max-docs 500 --budgets 12 16 32 64 128 --policies $POL || exit 1
python scripts/eval_policies.py --run blt_tinystories_entropy --max-docs 500 --budgets 12 16 32 64 128 --policies $POL --agg max --out policies_max.json || exit 1
python scripts/eval_policies.py --run blt_tinystories_fixed --max-docs 500 --budgets 12 16 32 64 128 --policies $POL || exit 1
python scripts/eval_policies.py --run blt_wikitext103_entropy --max-docs 60 --budgets 16 32 64 128 256 512 --policies $POL || exit 1
python scripts/eval_policies.py --run blt_wikitext103_fixed --max-docs 60 --budgets 16 32 64 128 256 512 --policies $POL || exit 1
