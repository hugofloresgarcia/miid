### vgg paper

#### some stuff I really found interesting/helpful

"such layers have a 7 × 7 effective receptive field. So what have we gained by using, for instance, a
stack of three 3×3 conv. layers instead of a single 7×7 layer? First, we incorporate three non-linear
rectification layers instead of a single one, which makes the decision function more discriminative.
Second, we decrease the number of parameters: assuming that both the input and the output of a
three-layer 3 × 3 convolution stack has C channels, the stack is parametrised by `3*(3**2 C**2) = 27C**2`
weights; at the same time, a single 7x7 conv. layer would require `7**2*C**2 = 49C**2` channels, 
81% more. This can be seen as imposing a regularisation on the 7 × 7 conv. filters, forcing them to
have a decomposition through the 3x3 filters (with non-linearity injected in between)"


"The incorporation of 1 × 1 conv. layers (configuration C, Table 1) is a way to increase the nonlinearity
of the decision function without affecting the receptive fields of the conv. layers. Even
though in our case the 1 × 1 convolution is essentially a linear projection onto the space of the same
dimensionality (the number of input and output channels is the same), an additional non-linearity is
introduced by the rectification function. It should be noted that 1×1 conv. layers have recently been
utilised in the “Network in Network” architecture of Lin et al. (2014)."