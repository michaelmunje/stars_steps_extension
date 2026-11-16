cd /scratch/09865/mmunje/stars-improve
python -m baselines.stars_ma_openloop --split test --datasets steps,ptdpes,madras,grand_central,ucy,eth,atc --cache-workers 32 --checkpoint social_s42=experiments/stars_ma/runs/si_final_social_s42/checkpoints/best.pt --output results/stars_improve_2026-11-15/test/final_social.json
