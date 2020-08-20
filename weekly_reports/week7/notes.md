### classifiers
I trained a linear SVM on the philarmonia dataset (without percussion). The dataset consists of 19 musical instrument classes.  

experiment output:

| exp_name | seed | preprocessor             | fischer_reweighting | pca_n_components | classifier 
|----------|------|--------------------------|---------------------|------------------|------------
| exp      | 42   | openl3-mel256-6144-music | False               | 512              | svm-linear 


metrics:

| metrics accuracy_score | metrics_precision  | metrics_recall     | metrics_f1        | 
|------------------------|--------------------|--------------------|-------------------| 
| 0.910864376880772      | 0.8894307492725116 | 0.8893196202437047 | 0.888798657613511 | 


the confusion matrix is [here](hugofloresgarcia.github.io/summer_milestones/runs/openl3_linear-svm_no-percussion/results/confusion_matrix.html)

the normalized confusion matrix is [here](hugofloresgarcia.github.io/summer_milestones/runs/openl3_linear-svm_no-percussion/results/confusion_matrix_normalized.html)

### DCASE papers

#### task 4: SED in domestic environments

Miyazaki, et. al won the DCASE 2020 task 4 (detection of sound events in domestic environments). Their top ranking models are a Conformer (Convolutional Transformer) and Transformer models. 

a link to their [DCASE technical report](http://dcase.community/documents/challenge2020/technical_reports/DCASE2020_Miyazaki_108.pdf)

a link to their [SED Transformer paper](https://doi.org/10.1109/ICASSP40776.2020.9053609)

a link to the original [conformer paper](https://arxiv.org/pdf/2005.08100.pdf) by Gulati et al.

Most systems I saw preprocessed their data using [mixup](http://arxiv.org/abs/1710.09412), among other things. 

#### task 5: Urban Audio Tagging