# things that I've done these past two weeks

## audacity labeler


## visualizing embeddings
I computed VGGish and OpenL3 embeddings and visualized the dataset using t-SNE. Look at the plots here:  

- [vggish](./figs/vggish_tsne.html)
- [openl3](./figs/openl3_tsne.html)


## comparing embeddings

ran 50 trials, comparing the preprocessor representation used. 

The task is to classify english horns from french horns. A 3-Nearest-Neighbor classifier was implemented. 

signal flow:   
- raw audio --> preprocessor --> PCA (components=all) --> fischer reweighing --> classifier

conditions tested:

- ised (with and without fischer reweighing)
- vggish(with and without fischer reweighing)
- openl3 model variants (with and without fischer reweighing)
	- input representation: mel128 or mel256
	- embedding sizes: 512 and 6144 dimensions
	- content type: environmental or music

number of training samples (per trial): 200
number of validation samples (per trial): 140

every trial, a subset of the dataset would be randomly sampled with a fixed random seed for all conditions. the seed number for each trial is stored in its .yaml config.

### accuracy
![emb_accuracy](./figs/embeddings_accuracy.png)

### precision
![emb_precision](./figs/embeddings_precision.png)

### recall
![emb_recall](./figs/embeddings_recall.png)

### f1 score
![emb_f1](./figs/embeddings_f1.png)

The OpenL3 variants show better performance than both vggish and ised. Additionally, all models show an improvement in performance after fischer reweighing. 


## comparing classifiers

ran 50 trials, comparing the classifier used. 

The task is to classify english horns from french horns. 

preprocessor: openl3-mel128-512-music variant

signal flow:  
raw audio --> preprocessor --> PCA (components=all) --> fischer reweighing --> classifier

conditions tested:

- K Nearest Neighbors 
	- K=3 
	- K=5
	- K=7
- SVM
	- RBF kernel
	- Linear kernel
	- Sigmoid kernel
	- Poly kernels (degree 2, 3, 4)

number of training samples (per trial): 200
number of validation samples (per trial): 140

every trial, a subset of the dataset would be randomly sampled with a fixed random seed for all conditions. the seed number for each trial is stored in its .yaml config.

### accuracy
![emb_accuracy](./figs/classifiers_accuracy.png)

### precision
![emb_precision](./figs/classifiers_precision.png)

### recall
![emb_recall](./figs/classifiers_recall.png)

### f1 
![emb_f1](./figs/classifiers_f1.png)


Linear SVM seems to take the cake here. 