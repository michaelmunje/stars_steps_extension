# SSL encoder flags per ladder encoder (sourced). E0 = L0 original STARS, E1 = +node motion features,
# E2 = +native 8-step rate, 24-NN/10 m graph, heading back-fill (still single-target graphs), E3 = per-window graphs.
DATA="--datasets steps,ptdpes,madras,grand_central"
E0="--edge-only --fully-connected --stars-grid 40 --stars-heading --target-slot --max-graph-nodes 18"
E1="--fully-connected --stars-grid 40 --stars-heading --target-slot --max-graph-nodes 18"
E2="--knn 24 --radius 10 --target-slot"
E3="--knn 24 --radius 10"
