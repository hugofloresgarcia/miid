# musical instrument labeling experiments

## week 8
scaper: https://github.com/justinsalamon/scaper

do overlap and add
100db difference between the loudest sound and every point for silence

similarity 
1. overlap and add
d
bow to deal with silence and overlap? 

todo for today:

1. figure out the overlap and add problem (MIL? but think about the frame overlapped segments that will get more screen time)
2. how to deal with silence? 
3. how to deal with silence between sounds?
4. try other embeddings (PANNs, MusiCNN)
5. train a neural net to get class probabilities. 

## usage

clone and install philharmonia dataset  
`git clone git@github.com:hugofloresgarcia/philharmonia-dataset`  
`pip3 install ./philharmonia-dataset`

download dataset audio files  
`python3 ./philharmonia-dataset/philharmonia_dataset/dl_dataset.py`

install requirements  
`pip3 install -r requirements.txt`

