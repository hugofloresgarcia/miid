### classifiers
I trained a linear SVM on the philarmonia dataset (without percussion). The model has already been integrated into the audacity labeler. 

experiment setup:


| seed | preprocessor    | fischer_reweighting | pca n_components | classifier |
|------|-----------------|---------------------|------------------|-----------|
| 42   | openl3-mel256-6144-music | False   | 512          | svm-linear |


metrics:

| metrics_accuracy| metrics_precision  | metrics_recall     | metrics_f1        | 
|------------------------|--------------------|--------------------|-------------------| 
| 0.910864376880772      | 0.8894307492725116 | 0.8893196202437047 | 0.888798657613511 | 


the confusion matrix is [here](https://hugofloresgarcia.github.io/summer_milestones/runs/openl3_svm-linear_no-percussion/results/confusion_matrix.html)

the normalized confusion matrix is [here](https://hugofloresgarcia.github.io/summer_milestones/runs/openl3_svm-linear_no-percussion/results/confusion_matrix_normalized.html)

I'm planning on training a neural net this weekend. 

--
The dataset is very clean (I'm pretty sure these audio samples are used in sample synths). All the audio files last anywhere between 0.5s and 5s. 



### DCASE papers
#### task 4: SED in domestic environments

Miyazaki, et al. won the DCASE 2020 task 4 (detection of sound events in domestic environments). Their top ranking models are a Conformer (Convolutional Transformer) and Transformer models. 

a link to their [DCASE technical report](http://dcase.community/documents/challenge2020/technical_reports/DCASE2020_Miyazaki_108.pdf)

a link to their [SED Transformer paper](https://doi.org/10.1109/ICASSP40776.2020.9053609)

a link to the original [conformer paper](https://arxiv.org/pdf/2005.08100.pdf) by Gulati et al. This paper is pretty dense (it assumes that you know all about transformers, depth-wise convolutions, [Swish activations](https://arxiv.org/pdf/1710.05941.pdf), among other things). 

--

Most systems I saw preprocessed their data using [mixup](http://arxiv.org/abs/1710.09412), among other things. [SpecAugment](https://arxiv.org/pdf/1904.08779.pdf) is also popular, though it seems to be more popular in the speech recognition community. 

Another recurring technique in the literature is to use [multiple instance learning](https://doi.org/10.1016/j.artint.2013.06.003), where frame level predictions are pooled to produce a weak label.

-

### task 5: Urban Sound Tagging with Spatiotemporal Context

Augustin Arnault et al. won task 5. A link to their technical report [here](http://dcase.community/documents/challenge2020/technical_reports/DCASE2020_Arnault_70_t5.pdf)

Their best system used three embeddings:

- a generic audio embedding pre-trained on [TalNet](https://arxiv.org/pdf/1810.09050.pdf). 
	- In the TALNet paper, the authors compare MIL pooling functions and conclude that linear softmax pooling seems to perform best. 

- a task-specific embedding (CNN -> Transformer)
- a metadata embedding (Spatiotemporal Context)

These embeddings are then concatenated and used as input for a fully connected layer. 

--
#### PANNs
Kong et al. introduced PANNs in 2019 (Pretrained Audio Neural Networks). They use pretrained embeddings trained on Audioset (like VGGish), and then fine tune them for other tasks. A link to the paper [here](https://arxiv.org/pdf/1912.10211.pdf)

Iqbal et al. got second place in task 5 with a mean [ensemble of PANNs](http://dcase.community/documents/challenge2020/technical_reports/DCASE2020_Iqbal_38_t5.pdf)