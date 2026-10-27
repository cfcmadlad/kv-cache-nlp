#!/usr/bin/env bash
cd "$(dirname "$0")/../.." || exit 1
export PYTHONPATH=.
[ -f data/bytes/pg19/test.bin ] || python scripts/prepare_pg19.py || exit 1
python scripts/prepare_pg19.py --verify || exit 1
[ -f data/bytes/pg19/entropy_entropy_wikitext103_mix/test.npy ] || python scripts/cache_entropy.py --run entropy_wikitext103_mix --dataset pg19 --splits test || exit 1
one() {
    [ -f runs/$1/stream_pg19_$2_$3.json ] || python scripts/eval_stream.py --run $1 --books $2 $3 --out stream_pg19_$2_$3.json
}
export -f one
for run in blt_wikitext103mix_entropy_x10 blt_wikitext103mix_entropy; do
    printf '%s\n' "$run 0 5" "$run 5 10" "$run 10 15" "$run 15 20" | xargs -P 4 -L 1 bash -c 'one "$0" "$1" "$2"' || exit 1
done
