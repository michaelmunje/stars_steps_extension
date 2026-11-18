#!/bin/bash
# Official AgentFormer test.py + eval.py per fold (CPU), via the alias shim.
# usage: agentformer_official.sh eth hotel univ zara1 zara2   (WORK dir layout: see src/forecast_bench/predict_agentformer.py)
A=/scratch/09865/mmunje/.cache/agentformer; R=/scratch/09865/mmunje/stars-forecast-benchmark/references/agentformer
cd $A/work && source /scratch/09865/mmunje/stars-forecast-benchmark/.venv/bin/activate
for f in "$@"; do
  python /scratch/09865/mmunje/stars-forecast-benchmark/scripts/forecast_bench_upstream/agentformer_run.py $R/test.py --cfg ${f}_agentformer --gpu -1 > $A/test_$f.log 2>&1
  d=$(ls -d results/${f}_agentformer/results/epoch_*/test/samples | tail -1)
  echo "== $f $d"; python /scratch/09865/mmunje/stars-forecast-benchmark/scripts/forecast_bench_upstream/agentformer_run.py $R/eval.py --dataset $f --results_dir $d --data test --log_file $A/eval_$f.log 2>&1 | grep -E "^[0-9]+ (ADE|FDE)"
done
