## Convolutional Recurrent Neural Networks for Polyphonic Sound Event Detection
### authors: Emre Çakir, Giambattista Parascandolo, Toni Heittola, Heikki Huttunen, and Tuomas Virtanen


### the problem with fully connected nets
- lacks time and frequency invariance -> fixed connections between input and hidden units allow to model small variations in events 
	- we can fix this with a CNN
- temporal context is restricted to short time windows, preventing effective modeling of longer events like rain, and correlations between different events
	- ah, ha! this is where an RNN would come in handy.

