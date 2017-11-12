# week2 - Computing Parameters Analytically

## Normal Equation

Method to solve $\theta$ analytically  

X: matrix of coefficients (including $x_0$)
y: vector of our outputs

$$ \theta = (X^T X)^{-1} X^T y$$

```Octave
pinv (X' * X) * X' * y
```

There is **no need** to use **Feature scaling** when dealing with **Normal Equation**

### Comparative

$m$ training examples, $n$ features

* Gradient Descent

    * Need to choose $\alpha$
    * Needs many iterations
    * Works weel even when $n$ is large (O $(kn^2)$)


* Normal Equation
    * No need to choose $\alpha$
    * Don't need to iterate
    * Need to compute $(X^T X)^{-1}$    (computing (n x n) matrix is O$(n^3)$ ), altough this matrix has shape m by (n+1)
    * Slow if $n$ is very large

With the normal equation, computing the inversion has complexity O(n3). So if we have a very large number of features, the normal equation will be slow. In practice, when n exceeds 10,000 it might be a good time to go from a normal solution to an iterative process.

## Normal Equation and non-invertibility (optional)

* What if $X^T X$ is non-invertible? (singular/degenerate)

Octave have two functions for inverting matrices

* **pinv** (pseudo-inverse) - this is always return a invert matrix
* **inv** (inverse)

The difference between the two advanced numerical computing

* **Redundant** features, where two features are very closely related (i.e. they are **linearly dependent**)
* **Too many features** (e.g. m ≤ n). In this case, delete some features or use "**regularization**" (to be explained in a later lesson).

If $X^T X$ is **noninvertible**, the common causes might be having :

Solutions to the above problems include deleting a feature that is linearly dependent with another or deleting one or more features when there are too many features.