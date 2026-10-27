#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1
for s in run_tsmix run_controls run_long run_attn_policies; do
    bash scripts/campaign/$s.sh || exit 1
done
