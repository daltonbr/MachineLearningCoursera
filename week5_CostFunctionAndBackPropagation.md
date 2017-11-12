# week 5 - Cost Function And Back Propagation

## Cost function

NN (Classification)

L  = total # of layers in network  
$S_l$ = # units (not counting bias unit) in layer $l$

**e.g.**
$S_1 = 3$, $S_2 = 5$, $S_3 = 5$, $S_4 = 4$  

### Binary Classification

y = 0 or 1  
$ S_l = 1 $ (the last layer)  
K = 1 (only one final output layer)

### Multi-class classification (K classes)

$ y \in \mathbb{R}^K $

**e.g.**

\[
    \begin{bmatrix}
        1 \\ 0 \\ 0 \\ 0 \\
    \end{bmatrix}
    ,
    \begin{bmatrix}
        0 \\ 1 \\ 0 \\ 0 \\
    \end{bmatrix}
    ,
    \begin{bmatrix}
        0 \\ 0 \\ 1 \\ 0 \\
    \end{bmatrix}
    ,
    \begin{bmatrix}
        0 \\ 0 \\ 0 \\ 1 \\
    \end{bmatrix}
\]

pedestrian, car, motorcycle, truck  (K = 4)  
K output units

Recall that in neural networks, we may have many output nodes. We denote $h_\Theta(x)_k$ as being a hypothesis that results in the $k^{th}$ output. Our cost function for neural networks is going to be a generalization of the one we used for logistic regression. Recall that the cost function for regularized logistic regression was:

$$ J(\theta) = - \frac{1}{m} \sum_{i=1}^m [ y^{(i)}\ \log (h_\theta (x^{(i)})) + (1 - y^{(i)})\ \log (1 - h_\theta(x^{(i)}))] + \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2 $$

For neural networks, it is going to be slightly more complicated:

\begin{gather*} J(\Theta) = - \frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K \left[y^{(i)}_k \log ((h_\Theta (x^{(i)}))_k) + (1 - y^{(i)}_k)\log (1 - (h_\Theta(x^{(i)}))_k)\right] + \frac{\lambda}{2m}\sum_{l=1}^{L-1} \sum_{i=1}^{s_l} \sum_{j=1}^{s_{l+1}} ( \Theta_{j,i}^{(l)})^2\end{gather*}

We have added a few nested summations to account for our multiple output nodes. In the first part of the equation, before the square brackets, we have an additional nested summation that loops through the number of output nodes.

In the regularization part, after the square brackets, we must account for multiple theta matrices. The number of columns in our current theta matrix is equal to the number of nodes in our current layer (including the bias unit). The number of rows in our current theta matrix is equal to the number of nodes in the next layer (excluding the bias unit). As before with logistic regression, we square every term.

Note:

* the double sum simply adds up the logistic regression costs calculated for each cell in the output layer
* the triple sum simply adds up the squares of all the individual Θs in the entire network.
* the i in the triple sum does **not** refer to training example i

## BackPropagation Algorithm

"**Backpropagation**" is neural-network terminology for **minimizing** our cost function, just like what we were doing with gradient descent in logistic and linear regression. Our goal is to compute:

$ \min_\Theta J(\Theta) $  

That is, we want to minimize our cost function J using an optimal set of parameters in theta. In this section we'll look at the equations we use to compute the partial derivative of $J(\theta)$ :

$ \dfrac{\partial}{\partial \Theta_{i,j}^{(l)}}J(\Theta)$  

# The Back propagation Algorithm

Given training set $ \lbrace (x^{(1)}, y^{(1)}) \cdots (x^{(m)}, y^{(m)})\rbrace $  

* Set $\Delta^{(l)}_{i,j}$ := 0 for all (l,i,j), (hence you end up having a matrix full of zeros)  

For training example t = 1 to m:

1. Set $a^(1):=x^{(t)}$
2. Perform forward propagation to compute $a^{(l)} for l=2,3,…,L$
3. Using $y^{(t)}$, compute $\delta^{(L)}=a^{(L)}−y^{(t)}$

Where L is our total number of layers and $a^{(l)}$ is the vector of outputs of the activation units for the last layer. So our "error values" for the last layer are simply the differences of our actual results in the last layer and the correct outputs in y. To get the delta values of the layers before the last layer, we can use an equation that steps us back from right to left:

4. Compute $\delta^{(L-1)}, \delta^{(L-2)},\dots,\delta^{(2)} $ using $\delta^{(l)} = ((\Theta^{(l)})^T \delta^{(l+1)})\ .*\ a^{(l)}\ .*\ (1 - a^{(l)})$

The delta values of layer l are calculated by multiplying the delta values in the next layer with the theta matrix of layer l. We then element-wise multiply that with a function called g', or g-prime, which is the derivative of the activation function g evaluated with the input values given by $z^{(l)}$.

The g-prime derivative terms can also be written out as:

$ g'(z^{(l)}) = a^{(l)}\ .*\ (1 - a^{(l)}) $

5. $ \Delta^{(l)}_{i,j} := \Delta^{(l)}_{i,j} + a_j^{(l)} \delta_i^{(l+1)} $ or with vectorization, $\Delta^{(l)} := \Delta^{(l)} + \delta^{(l+1)}(a^{(l)})^T$

Hence we update our new $ \Delta\ $ matrix.

$ D^{(l)}_{i,j} := \dfrac{1}{m}\left(\Delta^{(l)}_{i,j} + \lambda\Theta^{(l)}_{i,j}\right) $ , if j≠0.

$ D^{(l)}_{i,j} := \dfrac{1}{m}\Delta^{(l)}_{i,j} $, if j=0

The capital-delta matrix D is used as an "accumulator" to add up our values as we go along and eventually compute our partial derivative. Thus we get $ \frac \partial {\partial \Theta_{ij}^{(l)}} J(\Theta) = D_{ij}^{(l)} $

