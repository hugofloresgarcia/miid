K=8
(
for dir in ./experiments/classifiers_pca_3/seed_*; do
    ((i=i%K)); ((i++==0)) && wait
    python3 run_experiment.py -p $dir &
done
)

