import numpy as np
import torch
import torchaudio.transforms as transforms


def delta(x: torch.Tensor):
    """
    find delta values from MFCC
    """
    result = []
    for timestep in range(x.size()[1]-1):
        result.append((x[:, timestep] - x[:, timestep+1]).tolist())
        
    result = torch.Tensor(result).permute(1, 0)
    return result

def features(mfcc: torch.Tensor):
    """
    compute ISED feature vector from MFCCs
    """
    dmfcc = delta(mfcc)
    
    fmean = mfcc.mean(dim=-1)
    fvar = mfcc.var(dim=-1)
    
    dmean = dmfcc.mean(dim=-1)
    dvar = dmfcc.var(dim=-1)
    
    return torch.cat([fmean, fvar, dmean, dvar])
    
def distance(a, b):
    """
    euclidean distance
    """
    return np.sqrt(torch.sum((a-b)**2))

def relevance(s, sp, sn=None):
    """
    relevance score given positive and negative samples
    """
    if sn is None:
        return 1 / distance(s, sp)
    else:
        return distance(s, sn) / (distance(s, sn) + distance (s, sp))
    
    
def nearest_neighbor(sample: torch.Tensor, examples: torch.Tensor):
    if len(examples) == 0:
        return None
    # compute distances
    distances = []
    for e in examples:
        distances.append(distance(e, sample))
    
    # get the nearest neighbor's index
    idxs = np.argsort(np.array(distances))
    
    # the nearest neighbor will be the last index 
    return examples[idxs[-1]]
    
    
class ISEDmodel:
    def __init__(self,  target: torch.Tensor):
        """
        ISED model to hold weights, as well as positive and negative examples
        
        Parameters:
        target (torch.Tensor): our initial positive example (a feature vector)
        """
        self.target = target
        self.weights = torch.ones(target.size())
        self.examples = []
        
        self.examples.append({
            'features': target, 
            'label': 'positive'
        })
        
    def get_feature_map(self, label=None):
        """
        get weighted feature map of examples
        
        :param label: get positive or negative examples.
            if label is None, the complete feature map will be returned
        :return: torch.Tensor with feature map
        """
        if label == 'positive':
            m = self.get_subset(label, as_tensor=True)
            return torch.Tensor([self(e).tolist() for e in m])
        
        elif label == 'negative':
            m = self.get_subset(label, as_tensor=True)
            return torch.Tensor([self(e).tolist() for e in m])
        
        else:
            return torch.Tensor(
                [self(e['features']).tolist() for e in self.examples])
    
    def get_subset(self, label: str, as_tensor=False):
        """
        get unweighted feature map of all examples that 
        match label (positive or negative)
        
        :param label: (str)
        :param as_tensor: return torch.Tensor or list
        """
        if label is None:
            result = [e['features'].tolist() for e in self.examples]
        else:
            result = [e['features'].tolist() 
                  for e in self.examples if e['label'] == label]
        if as_tensor:
            return torch.Tensor(result)
        else:
            return result
    
    def add_example(self, f: torch.Tensor, label=str):
        """
        add new labeled example to model and compute it's relevance.

        """
        assert label in ('positive', 'negative'), "incorrect label"
        assert f.size() == self.weights.size(), "invalid f dimensions"
            
        # find our pos and neg neighbors
        sp, sn = self.find_neighbors(f)
        
        # compute the relevance of our example, in our weighed feature space
        rel = relevance(self(f), self(sp), self(sn))
        
        # store our example
        e = {
            'features': f,
            'label': label, 
            'relevance': rel
        }
        
        self.examples.append(e)
    
    def recompute_relevances(self):
        """
        compute relevances with updated feature weights
        """
        for i in range(len(self.examples)):
            # get our entry
            e = self.examples[i]
            f = e['features']
            
            # find nearest neighbors
            sp, sn, = self.find_neighbors(f)
            
            # compute relevance,  in weighted feature space
            rel = relevance(self(f), self(sp), self(sn))
            
            # store new relevance score
            e['relevance'] = rel
            
            self.examples[i] = e
    
    def reweigh_features(self):
        """
        reweigh our model weights using fischer's criterion
        """
        # make  2d tensors with features
        p = self.get_subset('positive')
        n = self.get_subset('negative')
        
        # we need more than 2 examples each to compute .std()? 
        if len(p) < 2 or len(n) < 2:
            return self

        # we are going to iterate per feature (variable),
        # so we must permute to iterate
        p = torch.Tensor(p)
        n = torch.Tensor(n)

        self.weights = (p.mean(dim=0)**2 - n.mean(dim=0)**2) / \
                            (p.std(dim=0)**2 + n.std(dim=0)**2)
        
        # compute all relevance scores in our updated feature space
        self.recompute_relevances()

        return self
        
    def find_neighbors(self, f: torch.Tensor):
        """
        find nearest positive and negative neighbors
        """
        # get a positive and negative subset of our examples
        positives = self.get_subset('positive', as_tensor=True)
        negatives = self.get_subset('negative', as_tensor=True)
        
        # multiply times our weight vector
        positives = self(positives)

        if not len(negatives) == 0:
            negatives = self(negatives)
        
        # get the nearest neighbors
        p = nearest_neighbor(f, positives)
        n = nearest_neighbor(f, negatives)
        
        return p, n
        
    def __call__(self, f: torch.Tensor):
        """
        multiply feature vector times model weights
        """
        if f is not None:
            return f * self.weights
        else: 
            return None
        
        

class PCA:
    def __init__(self):
        self.cov = None
        self.values = None
        self.vectors = None
    
    def fit(self, x: torch.Tensor):
         # find covariance
        self.cov = np.cov(x.numpy().T, rowvar=True)

        # eigendecomposition
        values, vectors = np.linalg.eig(self.cov)

        # sort 
        idx = np.argsort(values)[::-1]
        
        values = values[idx]
        vectors = vectors[:, idx]
        
        self.values = values
        self.vectors = vectors
        return self
    
    def get_covariance(self):
        return self.cov
    
    def transform(self, x: torch.Tensor, n_components: int):
        return torch.Tensor(
            x.numpy().dot(self.vectors[:, :n_components]))