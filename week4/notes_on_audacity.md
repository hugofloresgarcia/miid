## notes on building a labeler for audacity

**tl;dr**: I think going with Jack's way is the best option. However, we could simplify the code by loading VGGish through torch. If we load the VGGish model through torch, we don't have to use essentia anymore and can keep the dependencies to a minimum.  

### jack's AudacityLabeling:

- I got it to build fine a couple of times, but I (stupidly) updated another Audacity fork I had to the latest commit (which uses a newer version of wxWidgets than jack's), which messed up the build for AudadcityLabeling. I've been trying to fix this for the last couple of days. 
- Jack's fork of Audacity requires wxWidgets 3.1.1, which has been giving me a linker error every time I try to build. 
- The newest Audacity commit uses wxWidgets 3.1.3, which builds just fine. 
- I tried merging Jack's code into the newest Audacity commit, but several relevant variable names and function definitions have changed, which means that it may take some time. Moreover, the build system in the newest Audacity commit uses CMake, meaning I would have to add libtorch and essentia via CMake. 
	- torch has it's own CMake build script, while essentia uses `waf` to build, so I don't particularly know how I would add essentia to the CMake build. 
	

## options

### TorchScript and Extending Audacity Source

from the torch website:

>  We provide tools to incrementally transition a model from a pure Python program to a TorchScript program that can be run independently from Python, such as in a standalone C++ program. This makes it possible to train models in PyTorch using familiar tools in Python and then export the model via TorchScript to a production environment where Python programs may be disadvantageous for performance and multi-threading reasons.

**pros**

- this is what jack did
- Torch let's you load serialized models written in PyTorch in C++ using TorchScript. There are some limitations to this, but it can be an easy way of running experiments on a model in Python, and then serializing that same model for implementation in C++.


**cons**

- only nn.Module subclasses may be converted to TorchScript, which means that preprocessing steps would still have to be implemented in C++
- there are some limitations to what can be done inside the models. For example, adding control flow statements requires extra work that may end up overcomplicating things. If this is the case, I believe the model can be written using PyTorch's C++ frontend, and then restoring the model weights to a checkpoint. 

#### Using Audacity's mod-script-pipe

**update**: while adding labels may be easy, there is no way to pass the audio into python using this interface. I could see a way of using VamPy and mod-script-pipe together, but it seems overcomplicated. 

_from the audacity manual_:

> What Scripting can do

> Commands that Scripting uses are the same as in the Audacity macros feature. You can for example:

> - Select audio
- Apply effects
- Rearrange clips
- Export the results.
Scripting goes beyond the simpler tasks and presets of macros. Using Python scripting can for example do calculations about regions to select, or can make decisions on the basis of number and types of tracks in a project. It is also possible to build additional user interface in Python, for example an extra toolbar, and have it send commands to Audacity over the pipe.

**pros**

- python
- looks like adding labels at different points to a track is incredibly easy
- can do labeling, cutting and splitting clips, applying effects, and other things that may be useful in the future for eyes-free tools. 

**cons**

- For some menu commands, the project window must have focus for the command to succeed.
- limited commands available. full list of commands [here](https://manual.audacityteam.org/man/scripting_reference.html). 
- there are more warnings and known issures [here](https://manual.audacityteam.org/man/scripting.html#Known_Issues_.26_Missing_Features)


#### Using VamPy Plugins
**pros**

- write code in python!! spend less time figuring out that triple pointer and more time doing research!!
- there is also a vamp host for Sonic Visualizer, meaning that this will run in Sonic Visualizer too.

**cons**

- can process audio (mod-script-pipe can't). 
	- I don't think it can add labels to tracks :/. I went through the docs and it looks like everything has to be done through the plugin GUI, and it can't change add/remove labels in Audacity. 
- only supports python 2.7, which may break some things. I'm sure NumPy, Torch, and Tensorflow all work fine on python 2.7, though. 
- **update**: future versions of MacOS will definitely [break vampy](https://code.soundsoftware.ac.uk/issues/1897)



