import numpy as np
import torch 
import torchaudio
import datetime
import os
import yaml
import collections

#---------------------#
#     PLOT UTILS
#---------------------#
def smart_plotly_export(fig, save_path):
    img_format = save_path.split('.')[-1]
    if img_format == 'html':
        fig.write_html(save_path)
    elif img_format == 'bytes':
        return fig.to_image(format='png')
    #TODO: come back and make this prettier
    elif img_format == 'numpy':
        import io 
        from PIL import Image

        def plotly_fig2array(fig):
            #convert Plotly fig to  an array
            fig_bytes = fig.to_image(format="png", width=1200, height=700)
            buf = io.BytesIO(fig_bytes)
            img = Image.open(buf)
            return np.asarray(img)
        
        return plotly_fig2array(fig)
    elif img_format == 'jpeg' or 'png' or 'webp':
        fig.write_image(save_path)

def save_confusion_matrix(m, labels, save_path=None):
    import plotly.figure_factory as ff

    x = labels
    y = labels

    # change each element of z to type string for annotations
    m_text = [[str(y) for y in x] for x in m]

    # set up figure 
    fig = ff.create_annotated_heatmap(m, x=x, y=y, annotation_text=m_text, colorscale='Viridis')

    # add title
    fig.update_layout(title_text='<b>Confusion matrix</b>')

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))

    # add colorbar
    fig['data'][0]['showscale'] = True

    return smart_plotly_export(fig, save_path)
    
def dim_reduce(emb, labels, save_path, n_components=3, method='umap', title=''):
    """
    dimensionality reduction for visualization!
    saves an html plotly figure to save_path
    parameters:
        emb (np.ndarray): the samples to be reduces with shape (samples, features)
        labels (list): list of labels for embedding
        save_path (str): path where u wanna save ur figure
        method (str): umap, tsne, or pca
        title (str): title for ur figure
    returns:    
        proj (np.ndarray): projection vector with shape (samples, dimensions)
    """
    if method == 'umap':
        reducer = umap.UMAP(n_components=n_components)
    elif method == 'tsne':
        reducer = TSNE(n_components=n_components)
    elif method == 'pca':
        reducer = PCA(n_components=n_components)
    else:
        raise ValueError
 
    proj = reducer.fit_transform(emb)

    if n_components == 2:
        df = pd.DataFrame(dict(
            x=proj[:, 0],
            y=proj[:, 1],
            instrument=labels
        ))
        fig = px.scatter(df, x='x', y='y', color='instrument',
                        title=title_prefix+f"_{method}")

    elif n_components == 3:
        df = pd.DataFrame(dict(
            x=proj[:, 0],
            y=proj[:, 1],
            z=proj[:, 2],
            instrument=labels
        ))
        fig = px.scatter_3d(df, x='x', y='y', z='z',
                        color='instrument',
                        title=title)
    else:
        raise ValueError("cant plot more than 3 components")

    fig.update_traces(marker=dict(size=6,
                                  line=dict(width=1,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))

    return smart_plotly_export(fig, save_path)



#---------------------#
#     GENERAL UTILS
#---------------------#

def flatten_dict(d, parent_key='', sep='_'):
    """
    took this from
    https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def mkdir(path):
    if os.path.exists(path):
        return path

    os.makedirs(path)
    return path


def pretty_print(dictionary):
    data = yaml.dump(dictionary, default_flow_style=False)
    print(data)

def assert_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    elif isinstance(x, torch.Tensor):
        return x
    else:
        raise TypeError("input was neither an np array or tensor")

def assert_numpy(x):
    """
    make sure we're getting a numpy array (not a torch tensor)
    fix it if we need to
    """
    if isinstance(x, torch.Tensor):
        x = x.detach().numpy()
    else:
        x = x

    assert isinstance(x, np.ndarray), "ruh row, something went wrong converting to numpy"
    return x

def get_time():
    """
    return a date string w dir-compatible formatting (no spaces or slashes)
    """
    t = datetime.datetime.now()
    return ('_').join(t.strftime('%x').split('/')) + '_'+('.').join(t.strftime('%X').split(':'))


class Resample:
    def __init__(self, old_sr: int, sr: int):
        # resample 
        self.resample = torchaudio.transforms.Resample(old_sr, sr)
        
    def __call__(self, audio: torch.Tensor):
        audio = self.resample(audio)
        return audio