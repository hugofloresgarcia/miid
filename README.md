# hugo's summer milestones

## install requirements (python3)
`pip3 install -r requirements.txt`

## week 1 

### reading: Constant Q Paper
Notes are in: `readings/week1/const_q.md`

### coding skill
Show me a PCA dimensional reduction scatter plot of the ISED dimensions.  Take an audio file, split into 1-second chunks. Represent as ISED dimensions (based on MFCCs) and show a scatter plot where each point represents a single one-second chunk.  Ideally, this should be an easy command line thing you run (could be a Jupyter notebook or an Colab thing).

### coding deliverable: week1.py

usage: 

`python3 week1.py path_to_audio`

    positional arguments:
      path_to_audio         path to audio file to analyze

    optional arguments:
      -h, --help            show this help message and exit
      --sample_rate SAMPLE_RATE, -s SAMPLE_RATE
                        sample rate used for analysis
      --chunk_len CHUNK_LEN, -l CHUNK_LEN
                        chunk length of analysis, in seconds
      --n_mfcc N_MFCC       number of MFCCs
      --db_mels             use dB scaled spectrogram instead of log
      --n_fft N_FFT         fft window window size (in s)
      --no_normalization, -n
                        skip normalization of feature vectors to mean 0 and std 1
      --output_path OUTPUT_PATH, -o OUTPUT_PATH
                        path to save plots to`
                        
 ## week 2
 
 ### coding skill
 Now show me something where (1) there's a scatter plot with 1 second segments of audio, some of which are of sound class A and some are of sound class B.  When you do the PCA then scatter plot, we'll see that class A and class B are close to each other (2) Then, after providing some labels to the system for the points, it reweights the dimensions. (3) Then you can show a new scatter plot where the classes are fartehr apart.
 
 ### coding deliverable
 
 download the philharmonia samples first:
 
 `python3 dl_dataset.py`
 
 pick 2 or more classes from the philarmonia dataset
 
 `python3 list_classes.py`

 run the week2 script with desired classes (example):
 
 `python3 week2.py french-horn english-horn  --target english-horn`
 
 note: the output figures will be saved in ./week2_output
 
     positional arguments:
      classes               classes to obtain from dataset. run list_classes.py to see available classnames

    optional arguments:
      -h, --help            show this help message and exit
      --target TARGET, -t TARGET
                            target class. if none specified, first entry in classes will be used
      --log_every LOG_EVERY, -l LOG_EVERY
                            save an image to output dir every... (default: 30)
      --max_samples MAX_SAMPLES, -m MAX_SAMPLES
                            maximum number of samples to obtain from dataset (default: 300)
      --sample_rate SAMPLE_RATE, -s SAMPLE_RATE
                            sample rate used for analysis (default: 8000)
      --n_mfcc N_MFCC       number of MFCCs (default: 13)