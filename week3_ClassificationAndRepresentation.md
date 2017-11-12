# Week3 - Decision Boundary

In order to get our discrete 0 or 1 classification, we can translate the output of the hypothesis function as follows:

$ h_\theta(x) \geq 0.5 \rightarrow y = 1 $  
$ h_\theta(x) < 0.5 \rightarrow y = 0 $

The way our logistic function g behaves is that when its input is greater than or equal to zero, its output is greater than or equal to 0.5:

$ g(z) \geq 0.5 $  
$ when \; z \geq 0 $

Remember:  

$ z=0, e^{0}=1 \Rightarrow g(z)=1/2 $  
$ z \to \infty, e^{-\infty} \to 0 \Rightarrow g(z)=1 $  
$ z \to -\infty, e^{\infty}\to \infty \Rightarrow g(z)=0 $  

So if our input to g is $\theta^Tx$ , then that means:

$  h_\theta(x) = g(\theta^T x) \geq 0.5 $  
$ when \; \theta^T x \geq 0 $  

From these statements we can now say:

$ \theta^T x \geq 0 \Rightarrow y = 1 $  
$ \theta^T x < 0 \Rightarrow y = 0 $

The **decision boundary** is the line that separates the area where y = 0 and where y = 1. It is created by our hypothesis function.

$ \theta = \begin{bmatrix}5 \\ -1 \\ 0 \end{bmatrix} $  

$ y = 1 \; if \; 5 + (-1) x_1 + 0 x_2 \geq 0 $  
$ 5 - x_1 \geq 0 $  
$ - x_1 \geq -5 $  
$ x_1 \leq 5 $

In this case, our decision boundary is a straight vertical line placed on the graph where x1=5, and everything to the left of that denotes y = 1, while everything to the right denotes y = 0.

Again, the input to the sigmoid function g(z) (e.g. $\theta^Tx$) doesn't need to be linear, and could be a function that describes a circle (e.g. z=θ0+θ1x21+θ2x22) or any shape to fit our data.