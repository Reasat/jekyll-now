---
layout: post
title: "Practical lossless compression with latent variables using bits back coding"
mathjax: true
---
*James Townsend, Thomas Bird, David Barber*<br/>
https://arxiv.org/pdf/1901.04866.pdf

# Components of the compression pipeline

**VAE** used to encode the input to a latent space. This works as the first level of compression?
**BB-ANS** used to further compress the latent space.

# What is a VAE?

https://arxiv.org/pdf/1312.6114.pdf

https://arxiv.org/pdf/1606.05908.pdf

<img src="/assets/images/vae_model.png" alt="Variational Autoencoder Model">

## Setting up definitions

$X={\\{x^i\\}}^{N}_{i=1}$, is the set of data samples, $z$ is the latent variable 

Say we have a family of deterministic functions $f_{\theta}(z)$, parameterized by a vector $\theta$ in some space $\Theta$, where $f: \mathcal{Z}\times\Theta \to \mathcal{X}$.
$f$ is deterministic, but if $z$ is random and $\theta$ is fixed, then $f(z;\theta)$ is a random variable in the space $\mathcal{X}$.
We wish to optimize $\theta$ such that we can sample $z$ from $P(z)$ and, with high probability, $f(z;\theta)$ will be like the $X$'s in our dataset. 
 
$p_{\theta}(x)$ is the **marginal likelihood**, $p_{\theta}(x)=\int p_{\theta}(x\|z)p_{\theta}(z)dz$

$$p_{\theta}(z)$$ is the **prior**

$p_{\theta}(z\|x)$ is the **posterior likelihood** or the **probabilistic encoder**, e.g. a neural network with a non-linear hidden layer. 

$p_{\theta}(x\|z)$ is the **likelihood** function, the **decoder**, or the **generative** function. Typically another neural network. In VAEs, the choice of this output distribution is often Gaussian, i.e., $p_{\theta}(x|z) = \mathcal{N}(X|f_{\theta}(z),\sigma^2*I)$.
That is, it has mean $f_{\theta}(z)$ and covariance equal to the identity matrix $I$ times some scalar $\sigma$ (which is a hyperparameter).



## Why is $p_{\theta}(z\|x)$ computationally intractable?
<style>
.column {
  float: left;
  width: 33.33%;
  padding: 5px;
}

/* Clearfix (clear floats) */
.row::after {
  content: "";
  clear: both;
  display: table;
}
</style>


<div class="row">
  <div class="column">
    <img src="/assets/images/mnist_5_big.png" alt="Snow" style="width:100%">
  </div>
  <div class="column">
    <img src="/assets/images/mnist_5_b_big.png" alt="Forest" style="width:100%">
  </div>
  <div class="column">
    <img src="/assets/images/mnist_5_c_big.png" alt="Mountains" style="width:100%">
  </div>
</div>
Now all that remains is to maximize Equation~\ref{eq:total}, where $P(z)=\mathcal{N}(z|0,I)$.  
As is common in machine learning, if we can find a computable formula for $P(X)$, and we can take the gradient of that formula, then we can optimize the model using stochastic gradient ascent.
It is actually conceptually straightforward to compute $P(X)$ approximately: we first sample a large number of $z$ values $\{ z_{1} , ..., z_{n}\}$, and compute $P(X)\approx\frac{1}{n}\sum_i P(X|z_i)$.  
The problem here is that in high dimensional spaces, $n$ might need to be extremely large before we have an accurate estimate of $P(X)$.  
To see why, consider our example of handwritten digits.  
Say that our digit datapoints are stored in pixel space, in 28x28 images as shown in Figure~\ref{fig:digits}.
Since $P(X|z)$ is an isotropic Gaussian, the negative log probability of $X$ is proportional squared Euclidean distance between $f(z)$ and $X$.
Say that Figure~\ref{fig:digits}(a) is the target ($X$) for which we are trying to find $P(X)$.
A model which produces the image shown in Figure~\ref{fig:digits}(b) is probably a bad model, since this digit is not much like a 2.
Hence, we should set the $\sigma$ hyperparameter of our Gaussian distribution such that this kind of erroroneous digit does not contribute to $P(X)$.
On the other hand, a model which produces Figure~\ref{fig:digits}(c) (identical to $X$ but shifted down and to the right by half a pixel) might be a good model.
We would hope that this sample would contribute to  $P(X)$.
Unfortunately, however, we can't have it both ways: the squared distance between $X$ and Figure~\ref{fig:digits}(c) is .2693 (assuming pixels range between 0 and 1), but between $X$ and Figure~\ref{fig:digits}(b) it is just .0387.  
**The lesson here is that in order to reject samples like Figure~\ref{fig:digits}(b), we need to set $\sigma$ very small, such that the model needs to generate something *significantly* more like $X$ than Figure~\ref{fig:digits}(c)!
Even if our model is an accurate generator of digits, we would likely need to sample many thousands of digits before we produce a 2 that is sufficiently similar to the one in Figure~\ref{fig:digits}(a).**
We might solve this problem by using a better similarity metric, but in practice these are difficult to engineer in complex domains like vision, and they're difficult to train without labels that indicate which datapoints are similar to each other.
Instead, VAEs alter the sampling procedure to make it faster, without changing the similarity metric.

My thoughts: For each of the different $x_i \in X$, **we are computing  $i$ different distributions (mean and variance) through the encoder**. So during SGD for each $x_i$ we have to make sure that the decoder model sees enough randomly sampled $z$ so that they can be decoded to the output $x$'s to compute the loss function which will guide the encoder (and decoder) parameters to the right direction. But this becomes computationally challenging to optimize for each $x_i$ as the number of samples increase. 


## How does $q_{\phi}(z\|x)$ approximate $p_{\theta}(z\|x)$?



## How is a VAE different compared to a determenistic autoencoder?

## How does encoding\decoding with VAEs produce a lossless compression as stated in the paper?

## What is BB? What is ANS?

## How does BB+ANS interfacing work. Why is ANS preferred over arithmetic coding
