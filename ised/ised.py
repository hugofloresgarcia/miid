from .core import *
from .neighbors import *
from sklearn.decomposition import PCA

class Preprocessor:
    def __init__(self, sr, mfcc_kwargs, normalize=False):
        self.sr = sr
        self.mfcc_kwargs = mfcc_kwargs
        self.normalize = normalize

    def __call__(self, audio, sr):

        feats = compute_features(audio, sr, self.sr, self.mfcc_kwargs, self.normalize)

        return feats

class Model:
    def __init__(self, target: torch.Tensor, label: str):
        """
        ISED model to hold weight

        Parameters:
        target (torch.Tensor): our initial positive example (a feature vector)
        """
        target = utils.assert_numpy(target)
        self.target = {'features': target, 'label': label}
        self.weights = {label: np.ones(target.shape)}
        self.examples = []
        self.examples.append(self.target)

        self.pca = {}

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
        :param both: if true, both positive AND negative examples corresponding
            to that label will be returned
        :return: torch.Tensor with feature map
        """
        if both:
            # get ALL the examples
            m = self.get_subset(None, as_array=True)
            # multiply times our label weights
            return np.array([self(e, label).tolist() for e in m])
        else:
            m = self.get_subset(label, as_array=True, others=others)
            return np.array([self(e, label).tolist() for e in m])

    def get_subset(self, label: str, as_array=False, others=False):
        """
        get unweighted feature map of all examples that
        match label (positive or negative)

        :param label: (str)
        :param as_array: return torch.Tensor or list
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
        if as_array:
            return np.array(result)
        else:
            return result

    def add_example(self, f: np.ndarray, label=str):
        """
        add new labeled example to model and compute it's relevance.
        """
        assert f.shape == self.target['features'].shape, "invalid feature dimensions"
        if label not in self.weights:
            self.weights[label] = np.ones(self.target['features'].shape)

        e = {
            'features': f,
            'label': label,
        }

        self.examples.append(e)

        # add our relevance
        # e['relevance'] = self.relevance(e)
        # this is hacky but it will do for now
        # self.examples.pop()

        # self.examples.append(e)

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

            p = np.array(p)
            n = np.array(n)

            self.weights[label] = (p.mean(axis=0) ** 2 - n.mean(axis=0) ** 2) / \
                                  (p.std(axis=0) ** 2 + n.std(axis=0) ** 2)

            # VGGISH FIX
            # TODO: the last element in the vggish embedding is always 255. (why)
            # TODO: this breaks fischer's criterion because the std deviation
            # TODO: will always be 0, so you end up dividing by 0
            # TODO: I'm currently replacing the nan by 0 (meaning that the feature will
            # TODO: have no weight at all. should I be doing this?
            for i, w in enumerate(self.weights[label]):
                if np.isnan(w):
                    self.weights[label][i] = 0


            # compute all relevance scores in our updated feature space
            # we don't need relevances for now tho and the computation is slow
        #             self.recompute_relevances()

        return self

    def find_neighbors(self, e: dict):
        """
        find nearest positive and negative neighbors
        """
        # get a positive and negative subset of our examples
        me = self.get_subset(e['label'], as_array=True)
        others = self.get_subset(e['label'], as_array=True, others=True)

        # multiply times our weight vector
        me = self(me, e['label'])

        if not len(others) == 0:
            others = self(others, e['label'])

        # get the nearest neighbors
        p = get_neighbors(1, e['features'], me)
        n = get_neighbors(1, e['features'], others)

        return p, n

    def __call__(self, f: np.ndarray, label: str):
        """
        multiply feature vector times model weights
        """
        assert label in self.weights, "couldn't find weights for that label"
        f = utils.assert_numpy(f)
        w = self.weights[label]

        if f is not None:
            return f * w
        else:
            return None

    def do_pca(self, label, num_components, weights=True):
        """
        do PCA on the dataset

        params:
            label: label that corresponds to positive examples
            num_components: number of components to output
            weights: if true, the feature vectors will be multiplied times
                the model weights

        returns:
            tuple of the form (pmap, nmap) where
                pmap == positive examples (that correspond to the label)
                nmap == negative examples (that don't match the label)

        """
        if weights:
            # get the complete, positive and negative examples
            fmap = self.get_feature_map(label, both=True)
            nmap = self.get_feature_map(label, others=True)
            pmap = self.get_feature_map(label)

        else:
            fmap = self.get_subset(None, as_array=True)
            nmap = self.get_subset(label, others=True, as_array=True)
            pmap = self.get_subset(label, others=False, as_array=True)

        # now do PCA:
        pca = PCA(num_components)
        pca.fit(fmap)  # fit with both positive and negative

        self.pca[label] = pca
        # get transformed versions
        pmap = pca.transform(pmap)
        nmap = pca.transform(nmap)

        return pmap, nmap
