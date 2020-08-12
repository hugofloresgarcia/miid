K=4
(
for dir in ./experiments/embeddings_pca_16/seed_*; do
    ((i=i%K)); ((i++==0)) && wait
    python3 run_experiment.py -p $dir &
done
)


K=4
(
for dir in ./experiments/classifiers_pca_16/seed_*; do
    ((i=i%K)); ((i++==0)) && wait
    python3 run_experiment.py -p $dir &
done
)