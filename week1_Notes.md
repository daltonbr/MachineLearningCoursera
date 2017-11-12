# Machine Learning Course Notes - Week 1

## Introduction

A Stanford course by Andrew Ng - [ml-class.org](https://www.coursera.org/learn/machine-learning)  
Taught by:  **Andrew Ng**, Co-founder, Coursera; Adjunct Professor, Stanford University; formerly head of Baidu AI Group/Google Brain

"Machine Learning is the science of getting computers to learn, without being explicitly programmed."

Examples of the use of ML

* Ranking webpages at search engine
* Spam-filters
* Categorizing photos systems

**AI Dream**: to build machine as intelligent as humans
"Many AI researchers believe that the best way to towards that goal is through learning algorithms that try to mimic how the human brain learns." - Andrew Ng

Machine Learning

    * Grew out of work in AI
    * New capability for computers

### Examples
    * Database mining
        * Large datasets from growth of automation/web
        * E.g., Web click data, medical records, biology, engineering
    * Applications can't program by hand
        * E.g., Autonomous helicopter, handwriting recognition, most of Natural Language Processing (NLP), Computer Vision
    * Self-customized programs
        * E.g., Amazon, Netflix product recomendations
    * Understanding human learning (brain, real AI)

"A student showed me an article on the top twelve IT skills.
The skills that **IT hiring managers cannot say no to**. It was a slightly older article, but at the top of this list of the twelve most desirable IT skills was machine learning. Here at Stanford, the number of recruiters that contact me asking if I know any graduating machine learning students is far larger than the machine learning students we graduate each year. So I think there is a vast, unfulfilled demand for this skill set"

### What is Machine Learning?
Definitions:

* Arthur Samuel (1959). Machine Learning: Field of study that gives computers the ability to learn without being explicitly programmed.
* Tom Mitchell (1998) Well-posed Learning Problem: A computer program is said to *learn* from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.

#### Machine Learning Algorithms

* Supervised learning
* Unsupervised learning

Others: Reinforcement learning, recommender systems.

Also talk about: Practical advice for applying learnign algorithms.

### Supervised Learning
"right answers" given

**Regression**: Predict **continuous** valued output (e.g. price evaluation)
**Classification** **Discrete** valued output (e.g. 0 or 1)

### Unsupervised Learning
No labels, no "right answers"
Clustering algorithms
    * Organize computing clusters
    * Social network analysis
    * Market Segmentation
    * Astronomical data analysis
    * Cocktail party problem

Cocktail Party Problem algorithm

[W,s,v] = svd((repmat(sum(x.*x,1),size(x,1),1).*x)*x');

 * SVD function - stands for singular value decomposition; a linear algebra routine, that is just built into Octave.

## Model and Cost Function

**Training set** notation:

**m** = Number of training examples  
**x**'s = "input" variables / features  
**y**'s = "output" variables = "target" variables  

$ (x^{(i)}, y^{(i)}) $ - **i-th** training example

Training Set -> Learning Algorithm -> outputs a function **h** (by convention, from *hypothesis*)

E.g. (size_of_house *x*) -> h -> (estimated_price *y*)
$ h(x) = y $

*h* maps from x's tp y's

### How do we represent h (hypothesis)?

$$ h_\theta (x) = \theta_0 + \theta_1 x $$
Shorthand: $ h(x) $

We predict that y is a linear function of x

This is often called:  
Linear regression with one variable.  
Univariate linear regression.  

### Cost Function

$ \theta_i$'s are our parameters

**Idea**: Choose $ \theta_0, \theta_1 $ so that $ h_\theta (x) $ is close to $y$ for our training examples $(x, y)$

$$ J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta (x^{(i)}) - y^{(i)})^2 $$
m = #training examples  

$ h_\theta(x^{(i)})  = \theta_0 + \theta_1(x^{(i)}) $  

Goal: Minimize $ J(\theta_0 \theta_1) $  

Also called **squared error function**

### Gradient descent algorithm

Have some function $ J(\theta_0 \theta_1) $  (this is an arbitrary function, we can have many more parameters)  
We want the $ \theta_0, \theta_1 $ when J has min value.  

**Outline**

* Start with some $ \theta_0, \theta_1 $
* Keep changing $ \theta_0, \theta_1 $ to reduce J until we hopefully end up at a minimum

Repeat until convergence {  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$ \theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J (\theta_0, \theta_1) $  
}  
(for j = 0 and j = 1)  
$\alpha $ is our **learning rate**

* simultaneously update $ (\theta_0, \theta_1) $ - Because we use **both** to compute the partial derivative.  

**Correct update**

$ temp0 := \theta_0 - \alpha \frac{\partial}{\partial \theta_0} J (\theta_0, \theta_1) $  
$ temp1 := \theta_1 - \alpha \frac{\partial}{\partial \theta_1} J (\theta_0, \theta_1) $  
$ \theta_0 := temp0 $  
$ \theta_1 := temp1 $  

### Expanding the derivative term

Repeat until convergence {  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$ \theta_0 := \theta_0 - \alpha \frac{1}{m} \sum_{i=1}^m (h_\theta (x^{(i)}) - y^{(i)}) $  
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$ \theta_1 := \theta_1 - \alpha \frac{1}{m} \sum_{i=1}^m (h_\theta (x^{(i)}) - y^{(i)}) . x^{(i)} $  
}  

Since our cost function is a "convex function" (a "bowl-shape one) the gradient descent will always converge to global optimum, since all local optima are global.

Sometimes, this is also called **"Batch"** Gradient Descent  

"Batch": Each step of gradient descent uses all the training examples.