# Week 2 - Multivariate Linear Regression

Notation
$ m $ number of features  
$ n $ = number of features  
$ x^{(i)}$ input (features) of $i^{th}$ training example  
$ x^{(i)_j}$ value of feature j in th $i^{th}$ training example  

## Expanding our Hypothesis

$ h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n $  

For convenience of notation, define $x_0 = 1$  

$$
x = 
\begin{bmatrix} 
    x_0\\ x_1\\ x_2\\ \dots \\ x_n
\end{bmatrix} \in \mathbb{R}^{n+1}
$$

$$
\theta = 
\begin{bmatrix} 
    \theta_0\\ \theta_1\\ \theta_2\\ \dots \\ \theta_n
\end{bmatrix} \in \mathbb{R}^{n+1}
$$

$$ h_\theta(x) = \theta^T x $$  

$$ Multivariate Linear Regression $$

## New algorithm

Repeat until convergence {  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$ \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta) $ 
}  

Expanding the partial derivative  

Repeat until convergence {  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$ \theta_j := \theta_j - \alpha \frac{1}{m} \sum_{i=1}^m (h_\theta (x^{(i)}) - y^{(i)})x_{j}^{(i)} $ 
}  
(simultaneously update $\theta_j$ for $j = 0, ..., n$)

\begin{align*} & \text{repeat until convergence:} \; \lbrace \newline \; & \theta_0 := \theta_0 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_0^{(i)}\newline \; & \theta_1 := \theta_1 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_1^{(i)} \newline \; & \theta_2 := \theta_2 - \alpha \frac{1}{m} \sum\limits_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)}) \cdot x_2^{(i)} \newline & \cdots \newline \rbrace \end{align*}

## Practice I - Feature Scaling

**Idea**: Make sure features are on a similar scale
E.g.  
$x_1 = size (0-2000 feet^2)$  
$x_2 = number Of Bedrooms (1-5)$  

If we "normalize" or parameters, our countour plots will look like circles, so our algorithm will converge faster.

**Possible solution**  
$x_1 = size (0-2000 feet^2) / 2000$  
$x_2 = number of bedrooms (1-5) / 5$  

**More generally**
Get every feature into approximately $-1 \le x_i \le 1$ range  

**Other approach - Mean Normalization**  
Replace $x_i$ with $x_i - \mu_i$ to make features have approximately zero mean (Do not apply to $x_0 - 1$).

E.g.  
$x_1 = \frac{size - average_size}{2000}$  
$x_2 = \frac{bedrooms - average_bedrooms}{5}$

**In general**:  
$$x_n = \frac{x_n - \mu_n}{S_n}$$ where S_n is the value Range in the training set (could be used the Standart Deviation either)

## Pratice II - Learning Rate

* "Debugging": How to make sure gradient descent is working correctly
* How to choose learn rate $\alpha$

Plot the cost function $J(\theta)$ as Gradient Descent runs, using the number of iterations.  
$J(\theta)$ should decrease after every iteration

### Example automatic convergence test

Declare convergence if $J(\theta)$ decreases by less than $10^-3$(our $\epsilon$ ) in one iteration

If $J(\theta)$ is increasing, maybe we are "overshooting" the minimum. To correct this use smaller $\alpha$ values

* For sufficiently small $\alpha$, $J(\theta)$ should decrease on every iteration
* But if $\alpha$ is too small, gradient descent can be slow to converge

## Features and Polynomial Regression

We can improve our features and the form of our hypothesis function in a couple different ways.

We can combine multiple features into one. For example, we can combine x1 and x2 into a new feature x3 by taking x1⋅x2.

Our hypothesis function need not be linear (a straight line) if that does not fit the data well.

We can change the behavior or curve of our hypothesis function by making it a quadratic, cubic or square root function (or any other form).

To make it a square root function, we could do: $h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 \sqrt{x_1}$  
One important thing to keep in mind is, if you choose your features this way then feature scaling becomes very important.
