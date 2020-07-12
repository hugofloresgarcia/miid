import numpy as np
import torch
import torchaudio.transforms as transforms
from sklearn.decomposition import PCA


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
    def __init__(self,  target: torch.Tensor, label: str):
        """
        ISED model to hold weight
        
        Parameters:
        target (torch.Tensor): our initial positive example (a feature vector)
        """
        self.target = {'features': target,'label': label}
        self.weights = {label: torch.ones(target.size())}
        self.examples = []
        
        self.examples.append(self.target)
    
    def get_labels(self):
        """
        get a tuple of unique class labels
        """
        return tuple(set([e['label'] for e in self.examples]))
        
    def get_feature_map(self, label=None, others=False, both=False):
        """
        get weighted feature map of examples (using the weights corresponding
        to that label)
        
        :param label: label to retrieve
        :param others: if false, all the positive examples corresponding to that
            label will be returned. Else, negative examples will be returned
        :param all: if true, both positive AND negative examples corresponding 
            to that label will be returned
        :return: torch.Tensor with feature map
        """
        if both:
            # get ALL the examples
            m = self.get_subset(None, as_tensor=True) 
            # multiply times our label weights
            return torch.Tensor([self(e, label).tolist() for e in m])
        else:
            m = self.get_subset(label, as_tensor=True, others=others)
            return torch.Tensor([self(e, label).tolist() for e in m])

    
    def get_subset(self, label: str, as_tensor=False, others=False):
        """
        get unweighted feature map of all examples that 
        match label (positive or negative)
        
        :param label: (str)
        :param as_tensor: return torch.Tensor or list
        """
        if label is None:
            result = [e['features'].tolist() for e in self.examples]
        else:
            if not others:
                result = [e['features'].tolist()
                          for e in self.examples if e['label'] == label]
            else:
                result = [e['features'].tolist()
                          for e in self.examples if not e['label'] == label]
        if as_tensor:
            return torch.Tensor(result)
        else:
            return result
    
    def add_example(self, f: torch.Tensor, label=str):
        """
        add new labeled example to model and compute it's relevance.
        """
        assert f.size() == self.target['features'].size(), "invalid feature dimensions"
        if label not in self.weights:
            self.weights[label] = torch.ones(self.target['features'].size())
    
        e = {
            'features': f,
            'label': label, 
        }
        
        self.examples.append(e)
        
        # add our relevance
        e['relevance'] = self.relevance(e)
        # this is hacky but it will do for now
        self.examples.pop()
        
        self.examples.append(e)
        
    def relevance(self, e):
        r = {}
        for label in self.get_labels():
                # find nearest neighbors
                sp, sn, = self.find_neighbors(e)
            
                # compute relevance,  in weighted feature space
                rel = relevance(self(e['features'], label), self(sp, label), self(sn, label))
            
                # store new relevance score
                r[label] = rel
        return r
    
    def recompute_relevances(self):
        """
        compute relevances with updated feature weights
        """
        for i in range(len(self.examples)):
            # get our entry
            e = self.examples[i]
            f = e['features']
            
            # we don't need relevances for now
            # e['relevance'] = self.relevance(e)
            
            self.examples[i] = e
    
    def reweigh_features(self):
        """
        reweigh our model weights using fischer's criterion
        """
        for label in self.get_labels():
            # make  2d tensors with features
            p = self.get_subset(label)
            n = self.get_subset(label, others=True)

            # we need more than 2 examples each to compute .std()? 
            if len(p) < 2 or len(n) < 2:
                return self

            p = torch.Tensor(p)
            n = torch.Tensor(n)

            self.weights[label] = (p.mean(dim=0)**2 - n.mean(dim=0)**2) / \
                                (p.std(dim=0)**2 + n.std(dim=0)**2)

            # compute all relevance scores in our updated feature space
            # we don't need relevances for now tho and the computation is slow
#             self.recompute_relevances()

        return self
        
    def find_neighbors(self, e: dict):
        """
        find nearest positive and negative neighbors
        """
        # get a positive and negative subset of our examples
        me = self.get_subset(e['label'], as_tensor=True)
        others = self.get_subset(e['label'], as_tensor=True, others=True)
        
        # multiply times our weight vector
        me = self(me, e['label'])

        if not len(others) == 0:
            others = self(others, e['label'])
        
        # get the nearest neighbors
        p = nearest_neighbor(e['features'], me)
        n = nearest_neighbor(e['features'], others)
        
        return p, n
        
    def __call__(self, f: torch.Tensor, label: str):
        """
        multiply feature vector times model weights
        """
        assert label in self.weights, "couldn't find weights for that label"
        w = self.weights[label]
        
        if f is not None:
            return f * w
        else: 
            return None