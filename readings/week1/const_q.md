## constant Q paper
### Introduction
The paper talks about a transform with a constant Q (center freq / bandwidth). The transform spaces 24 frequency bins evenly across the octave.

### Why not just plot a normal fft against log(f)?
This results in a really low resolution in the lower frequencies and an unnecesarily high resolution in the higher frequencies.

The paper gives an example of this. If you were to sample at 32kHz with a window of 1024 samples, you would have a frequency resolution of 31.3 Hz per bin. 

If you recorded a G3 (196 Hz) and a G\#3 (207.65 Hz), the frequency resolution wouldn't be high enough for you to tell them apart (33.3 Hz per bin vs. 11Hz difference in pitch).

### The Q value for the constant Q transform
```
bins_per_octave = 24
resolution_ratio = 2 ** (1/bins_per_octave) - 1 
resolution = lambda f: resolution_ratio * f

# Q = f / df 

so, for any value of f:

Q = lambda f: f / resolution(f)

# for a quartertone
# Q = f / (0.03 * f)
# Q = 34
```
this transform is also equivalent to a 1/24 filter/octave bank


### cool previous work
CCRMA --> Bounded Q Transform

1. calculate an FFT, discard frequency examples, just keep top octave
2. filter, downsample by a factor of 2
3. rinse and repeate with the filtered and downsampled audio

advantages: FFT, so fast

Kronland-Martinet -->  wavelet transform for musical analysis

- uses wavelets as basis functions
- "does not have sufficient resolution for note identification"


### calculation

consider the DFT (a really slow one):

```
import numpy as np

x = np.random.rand(1024) # time domain
W = np.hamming(x.shape) # hamming window
def dft(x):
    X = np.zeros(x.shape, dtype=np.complex64) # freq domain
    N = len(x) # window size
    for k in range(len(X)):
        for n in range(N):
            freq = -1j*2*np.pi * n / N
            X[k] += W[n] * x[n] * np.exp(freq)
```

in order to have a constant Q, we must vary the window size, 
so N must become N[k]. The frequency component becomes

```
freq = -j*2*np.pi * Q * n / N[k]
```

and our window function is
```
N[k] = (sr / f(k)) * Q
```
where `sr` is the sample rate, and `f(k)` the kth frequency of analysis 

Having a varying window length means that we must also have a varying window function, as it needs to keep the same shape for each frequency component. The new window function becomes a function of both the frequency component and the time-domain index, W[k, n].

Another cool thing to note is that you will always analyze the same amount of periods for all frequencies, since you're varying the time window.

note: the constant q transform is not invertible because
1. the number of samples betweeen calculations is greater than the analysis window length for the high-frequency bins. (i don't really understand what this means?)
2. the bandwidth is less than the frequency sampling interval for the bins where Q=68

Finally, the paper shows a set of violin examples, comparing the DFT to the Constant Q Transform. 

In these figures, the Constant Q Transform is much easier to visualize, since harmonics are spaced equally regardless of starting pitch. 