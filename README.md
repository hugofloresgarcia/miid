# musical instrument labeling experiments

## weeks 7
notes on /weekly_reports/week7/notes.md

## usage

clone and install philharmonia dataset  
`git clone git@github.com:hugofloresgarcia/philharmonia-dataset`  
`pip3 install ./philharmonia-dataset`

download dataset audio files  
`python3 ./philharmonia-dataset/philharmonia_dataset/dl_dataset.py`

install requirements  
`pip3 install -r requirements.txt`

generate trial configs  
`python3 generate_experiment.py -o path/to/my/experiment -g seed`

run experiment  
`python3 run_experiment.py -p path/to/my/experiment`

generate boxplots and other analyses  
`python3 compare_trials.py -p path/to/my/experiment -f independent_variable -m metrics_accuracy_score metrics_f1`

