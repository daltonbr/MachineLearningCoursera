# week3 - Logistic Regression

**Logistic Regression** is one of the most popular and used Linear Algorithms used today. It's a **classification** algorithm.

**Classification**

* Email Spam / Not Spam?
* Online Transaction Fraudulent (yes/no) ?
* Tumor: Malignant / Benign ?

$ y \in {0, 1} $

0: "Negative class"
1: "Positive class"

Later we will se multiclass classification problem.

Applying Linear Regression in classification problems isn't often a good idea

In logistic Regression $ 0 \le h_\theta(x) \le 1 $

Our new form uses the "Sigmoid Function," also called the "Logistic Function":

$ h_\theta (x) = g ( \theta^T x ) $

$ z = \theta^T x $

$ g(z) = \dfrac{1}{1 + e^{-z}} $