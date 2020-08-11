## notes on OpenL3 embedding
I tried some dimensionality reduction techniques on the OpenL3 embedding. Here are some results:

In the t-SNE and UMAP plots, the blue dots represent english horns, while the red dots represent french horns. 

### with fischer reweighing
#### PCA 
![l3pca](../experiments/openl3/exp_0/english-horn_validation.png)
#### t-SNE
![l3sne](../experiments/openl3/exp_0/t-sne_english-horn_validation.png)
#### UMAP
![l3umap](../experiments/openl3/exp_0/umap_english-horn_validation.png)

### without fischer reweighing
#### PCA 
![l3pca](../experiments/openl3/exp_1/english-horn_validation.png)
#### t-SNE
![l3sne](../experiments/openl3/exp_1/t-sne_english-horn_validation.png)
#### UMAP
![l3umap](../experiments/openl3/exp_1/umap_english-horn_validation.png)