{% include script.html %}
---
layout: post
title: "Practical lossless compression with latent variables using bits back coding"
mathjax: true
---
*James Townsend, Thomas Bird, David Barber*<br/>
https://arxiv.org/pdf/1901.04866.pdf

In this paper a **variational autoencoder (VAE)** is used to encode the input to a latent space. Then a **bits back coding scheme with asymetric numeral system compression
(BB-ANS)** is used to code the latent space (along with the likelihood distribution) to get the final form of the compression.

In this post, I am listing all the questions that came to mind while looking up VAEs and the BB-ANS compression scheme. Most of the text is directly taken from reference papers and may be paraphrased in some places to make it more coherent. Since the text is taken from different papers there may notational inconsistencies. I shall try to sort them out as I get a better understanding of the whole system.

## Setting up variables and probabilities
(Taken from [2])

$X={\\{x^i\\}}^{N}_{i=1}$, is the set of data samples, $z$ is the latent variable. In VAE's each dimension of $z$ is modeled as a zero mean unit variance gaussian distribution (i.e., $z \sim \mathcal{N}(0,I)$).

Say, we have a family of deterministic functions $f_{\theta}(z)$, parameterized by a vector $\theta$ in some space $\Theta$, where $f: \mathcal{Z}\times\Theta \to \mathcal{X}$.
$f$ is deterministic, but if $z$ is random and $\theta$ is fixed, then $f(z;\theta)$ is a random variable in the space $\mathcal{X}$.
We wish to optimize $\theta$ such that we can sample $z$ from $P_{\theta}(z)$ and, with high probability, $f(z;\theta)$ will be like the $X$'s in our dataset. 
 
$P_{\theta}(X)$ is the **marginal likelihood**, $P_{\theta}(X)=\int P_{\theta}(X\|z)P_{\theta}(z)dz$

$$P_{\theta}(z)$$ is the **prior**. In VAE's prior has a normal gaussian distribution, i.e., $P(z)=\mathcal{N}(0,I)$

$P_{\theta}(z\|x)$ is the **posterior likelihood** or the **probabilistic encoder**, e.g. a neural network with a non-linear hidden layer. 

$P_{\theta}(X\|z)$ is the **likelihood** function, the **decoder**, or the **generative** function. Typically another neural network. In VAEs, the choice of this output distribution is often Gaussian, i.e., $P_{\theta}(X|z) = \mathcal{N}(f_{\theta}(z),\sigma^2*I)$.
That is, it has mean $f_{\theta}(z)$ and covariance equal to the identity matrix $I$ times some scalar $\sigma$ (which is a hyperparameter). **But in the BB-ANS paper [4] they have modeled the pixel distributions with either a Bernoulli distribution (when dealing with binary MNIST) or a Beta-Binomial distribution (when dealing with uin8 MNIST)**.

**Note: We may or may not drop the $\theta$ subscript from the probabilities in the following sections.**

## What is a VAE?

(Taken from [2])

<img src="/assets/images/graphical_model.png" alt="" style="width:40%">
<figcaption> Figure 1: The standard VAE model represented as a graphical model.  
Note the conspicuous lack of any structure or even an "encoder" pathway: it is possible to sample from the model without any input. Here, the rectangle is "plate notation" meaning that we can sample from $z$ and $X$ $N$ times while the model parameters $\theta$ remain fixed.</figcaption>
(*I am not sure if this is the complete symbolic figure of the VAE since it does not show the decoder. Please refer to [1] for another variation of the figure*)

We are aiming maximize the probability of each $X$ in the training set under the entire generative process, according to:

$P_{\theta}(X)=\int P_{\theta}(X\|z)P_{\theta}(z)dz$&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;--------&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(1)

This framework, called "maximum likelihood", is used in VAEs.

## Why is $P_{\theta}(X\|z)$ (and also ($P_{\theta}(z\|X)$)) computationally intractable?

Taken from [2]

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
    <img src="/assets/images/mnist_5_big.png" alt="" style="width:100%">
  </div>
  <div class="column">
    <img src="/assets/images/mnist_5_b_big.png" alt="" style="width:100%">
  </div>
  <div class="column">
    <img src="/assets/images/mnist_5_c_big.png" alt="" style="width:100%">
  </div>
</div>
<figcaption>Figure 2: It's hard to measure the likelihood of images under a model using only sampling. 
Given an image $X$ (a), the middle sample (b) is much closer in Euclidean distance than the one on the right (c).
Because pixel distance is so different from perceptual distance, a sample needs to be extremely close in pixel distance to a datapoint $X$ before it can be considered evidence that $X$ is likely under the model.</figcaption>
Now all that remains is to maximize Equation 1, where $P(z)=\mathcal{N}(z|0,I)$. 
As is common in machine learning, if we can find a computable formula for $P(X)$, and we can take the gradient of that formula, then we can optimize the model using stochastic gradient ascent.
It is actually conceptually straightforward to compute $P(X)$ approximately: we first sample a large number of $z$ values $\{ z_{1} , ..., z_{n}\}$, and compute $P(X)\approx\frac{1}{n}\sum_i P(X|z_i)$. 
The problem here is that in high dimensional spaces, $n$ might need to be extremely large before we have an accurate estimate of $P(X)$. 
To see why, consider our example of handwritten digits. 
Say that our digit datapoints are stored in pixel space, in 28x28 images as shown in  Figure 2.
Since $P(X|z)$ is an isotropic Gaussian, the negative log probability of $X$ is proportional squared Euclidean distance between $f(z)$ and $X$.
Say that  Figure 2(a) is the target ($X$) for which we are trying to find $P(X)$.
A model which produces the image shown in  Figure 2(b) is probably a bad model, since this digit is not much like a 2.
Hence, we should set the $\sigma$ hyperparameter of our Gaussian distribution such that this kind of erroroneous digit does not contribute to $P(X)$.
On the other hand, a model which produces  Figure 2(c) (identical to $X$ but shifted down and to the right by half a pixel) might be a good model.
We would hope that this sample would contribute to  $P(X)$.
Unfortunately, however, we can't have it both ways: the squared distance between $X$ and  Figure 2(c) is .2693 (assuming pixels range between 0 and 1), but between $X$ and  Figure 2(b) it is just .0387. 
**The lesson here is that in order to reject samples like  Figure 2(b), we need to set $\sigma$ very small, such that the model needs to generate something *significantly* more like $X$ than  Figure 2(c)!
Even if our model is an accurate generator of digits, we would likely need to sample many thousands of digits before we produce a 2 that is sufficiently similar to the one in  Figure 2(a).**
We might solve this problem by using a better similarity metric, but in practice these are difficult to engineer in complex domains like vision, and they're difficult to train without labels that indicate which datapoints are similar to each other.
Instead, VAEs alter the sampling procedure to make it faster, without changing the similarity metric.

My thoughts: For each of the different $x_i \in X$, **we are computing  $i$ different distributions through the encoder**. So during SGD for each $x_i$ we have to make sure that the decoder model sees enough randomly sampled $z$ so that they can be decoded to the output $x$'s to compute the loss function which will guide the encoder (and decoder) parameters to the right direction. But this becomes computationally challenging to optimize for each $x_i$ as the number of samples increase. 


## How does $Q_{\phi}(z\|x)$ approximate $P_{\theta}(z\|x)$?

This means that we construct a new function $Q(z|X)$ which can take a value of $X$ and give us a distribution over $z$ values that are likely to produce $X$. 
Hopefully the space of $z$ values that are likely under $Q$ will be much smaller than the space of all $z$'s that are likely under the prior $P(z)$.
This lets us, for example, compute $E_{z\sim Q}P(X|z)$ relatively easily. (Here, $z \sim Q$ symbol means that a sample $z$ is drawn from $Q$ and $E(*)$ is the expectation operator.)
The relationship between $E_{z\sim Q}P(X|z)$ and $P(X)$ is one of the cornerstones of variational Bayesian methods.
We begin with the definition of Kullback-Leibler divergence (KL divergence or $\mathcal{D}$) between $P(z|X)$ and $Q(z)$, for some arbitrary $Q$ (which may or may not depend on $X$):
\begin{equation}
    \mathcal{D}\left[Q(z)\|P(z|X)\right]=E_{z\sim Q}\left[\log Q(z) - \log P(z|X) \right].
\label{eq:kl}
\end{equation}
We can get both $P(X)$ and $P(X|z)$ into this equation by applying Bayes rule to $P(z|X)$, (i.e. $P(z|X) = \frac{P(X,z)}{P(X)} = \frac{P(X\|z)P(z)}{P(X)}$):
\begin{equation}
    \mathcal{D}\left[Q(z)\|P(z|X)\right]=E_{z\sim Q}\left[\log Q(z) - \log P(X|z) - \log P(z) \right] + \log P(X).
\end{equation}
Here, $\log P(X)$ comes out of the expectation because it does not depend on $z$.  Negating both sides, rearranging, and contracting part of $E_{z\sim Q}$ into a KL-divergence terms yields:
\begin{equation}
    \log P(X) - \mathcal{D}\left[Q(z)\|P(z|X)\right]=E_{z\sim Q}\left[\log P(X|z)  \right] - \mathcal{D}\left[Q(z)\|P(z)\right].
\end{equation}

Note that $X$ is fixed, and $Q$ can be *any* distribution, not just a distribution which does a good job mapping $X$ to the $z$'s that can produce $X$.  
Since we're interested in inferring $P(X)$, it makes sense to construct a $Q$ which *does* depend on $X$, and in particular, one which makes $\mathcal{D}\left[Q(z)\||P(z|X)\right]$ small:
\begin{equation}
    \log P(X) - \mathcal{D}\left[Q(z|X)\|P(z|X)\right]=E_{z\sim Q}\left[\log P(X|z)  \right] - \mathcal{D}\left[Q(z|X)\|P(z)\right].
    \label{eq:variational}
\end{equation}

<div style="text-align: right"> --------&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(2)</div>

In two sentences, the left hand side has the quantity we want to maximize: $\log P(X)$ (plus an error term, which makes $Q$ produce $z$'s that can reproduce a given $X$; this term will become small if $Q$ is high-capacity).
The right hand side is something we can optimize via stochastic gradient descent given the right choice of $Q$ (although it may not be obvious yet how).

Now for a bit more detail on Equation 2.
Starting with the left hand side, we are maximizing $\log P(X)$ while simultaneously minimizing $\mathcal{D}\left[Q(z|X)\||P(z|X)\right]$. 
$P(z|X)$ is not something we can compute analytically: it describes the values of $z$ that are likely to give rise to a sample like $X$ under our model in Figure `. 
However, the second term on the left is pulling $Q(z|x)$ to match $P(z|X)$.
Assuming we use an arbitrarily high-capacity model for $Q(z|x)$, then $Q(z|x)$ will hopefully actually match $P(z|X)$, in which case this KL-divergence term will be zero, and we will be directly optimizing $\log P(X)$. 
As an added bonus, we have made the intractable $P(z|X)$ tractable: we can just use $Q(z|x)$ to compute it.


## How do we train the encoder and decoder networks that estimates probability distributions?
<img src="/assets/images/net.jpg" alt="" style="width:100%">
<figcaption>Figure 3: A training-time variational autoencoder implemented as a feedforward neural network, where $P(X|z)$ is Gaussian.  Left is without the "reparameterization trick", and right is with it.  Red shows sampling operations that are non-differentiable.  Blue shows loss layers.  The feedforward behavior of these networks is identical, but backpropagation can be applied only to the right network.</figcaption>

So how can we perform stochastic gradient descent on the right hand side of Equation 2?
First we need to be a bit more specific about the form that $Q(z|X)$ will take.
The usual choice is to say that $Q(z|X)=\mathcal{N}(z|\mu(X;\vartheta),\Sigma(X;\vartheta))$, where $\mu$ and $\Sigma$ are arbitrary deterministic functions with parameters $\vartheta$ that can be learned from data (we will omit $\vartheta$ in later equations).
In practice, $\mu$ and $\Sigma$ are again implemented via neural networks, and $\Sigma$ is constrained to be a diagonal matrix.
The advantages of this choice are computational, as they make it clear how to compute the right hand side.
The last term---$\mathcal{D}\left[Q(z|X)\|P(z)\right]$---is now a KL-divergence between two multivariate Gaussian distributions, which can be computed in closed form as:


$ \mathcal{D}[\mathcal{N}(\mu_0,\Sigma_0) \| \mathcal{N}(\mu_1,\Sigma_1)] = \frac{ 1 }{ 2 } \left( \mathrm{tr} \left( \Sigma_1^{-1} \Sigma_0 \right) + \left( \mu_1 - \mu_0\right)^\top \Sigma_1^{-1} ( \mu_1 - \mu_0 ) - k + \log \left( \frac{ \det \Sigma_1 }{ \det \Sigma_0  } \right)  \right)$
where $k$ is the dimensionality of the distribution.  In our case, this simplifies to:

$ \mathcal{D}[\mathcal{N}(\mu(X),\Sigma(X)) \| \mathcal{N}(0,I)] = \\ \frac{ 1 }{ 2 } \left( \mathrm{tr} \left( \Sigma(X) \right) + \left( \mu(X)\right)^\top ( \mu(X) ) - k - \log\det\left(  \Sigma(X)  \right)  \right).$
 
The first term on the right hand side of Equation 2 is a bit more tricky.
We could use sampling to estimate $E_{z\sim Q}\left[\log P(X|z)  \right]$, but getting a good estimate would require passing many samples of $z$ through $f$, which would be expensive.
Hence, as is standard in stochastic gradient descent, we take one sample of $z$ and treat $P(X|z)$ for that $z$ as an approximation of $E_{z\sim Q}\left[\log P(X|z)  \right]$.
After all, we are already doing stochastic gradient descent over different values of $X$ sampled from a dataset $D$.

The full equation we want to optimize is:
   $ E_{X\sim D}\left[\log P(X) - \mathcal{D}\left[Q(z|X)\|P(z|X)\right]\right]=E_{X\sim D}\left[E_{z\sim Q}\left[\log P(X|z)  \right] - \mathcal{D}\left[Q(z|X)\|P(z)\right]\right]$
<div style="text-align: right"> --------&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(3)</div>
If we take the gradient of this equation, the gradient symbol can be moved into the expectations.
Therefore, we can sample a single value of $X$ and a single value of $z$ from the distribution $Q(z|X)$, and compute the gradient of:
\begin{equation}
 \log P(X|z)-\mathcal{D}\left[Q(z|X)\|P(z)\right].
  \label{eq:onesamp}
\end{equation}
<div style="text-align: right"> --------&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(4)</div>
We can then average the gradient of this function over arbitrarily many samples of $X$ and $z$, and the result converges to the gradient of Equation 3.

There is, however, a significant problem with Equation 4.
$E_{z\sim Q}\left[\log P(X|z)  \right]$ depends not just on the parameters of $P$, but also on the parameters of $Q$.
However, in Equation 4, this dependency has disappeared!
In order to make VAEs work, it's essential to drive $Q$ to produce codes for $X$ that $P$ can reliably decode. 
To see the problem a different way, the network described in Equation 4 is much like the network shown in Figure 3 (left).
The forward pass of this network works fine and, if the output is averaged over many samples of $X$ and $z$, produces the correct expected value.
However, we need to back-propagate the error through a layer that samples $z$ from $Q(z|X)$, which is a non-continuous operation and has no gradient.
Stochastic gradient descent via backpropagation can handle stochastic inputs, but not stochastic units within the network!
The solution, called the "reparameterization trick" in [1], is to move the sampling to an input layer.
Thus, the equation we actually take the gradient of is:

$E_{X \sim D}\left[E_{\epsilon\sim\mathcal{N}(0,I)}[\log P(X\|z=\mu(X)+\Sigma^{1/2}(X)*\epsilon)]-\mathcal{D}\left[Q(z\|X)\|P(z)\right]\right]$
<div style="text-align: right"> --------&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(5)</div>

This is shown schematically in Figure 4 (right). 
Note that none of the expectations are with respect to distributions that depend on our model parameters, so we can safely move a gradient symbol into them while maintaning equality.
That is, given a fixed $X$ and $\epsilon$, this function is deterministic and continuous in the parameters of $P$ and $Q$, meaning backpropagation can compute a gradient that will work for stochastic gradient descent.

## How is testing done?

<img src="/assets/images/test_time.jpg" alt="" style="width:30%">
<figcaption>Figure 4: The testing-time variational "autoencoder", which allows us to generate new samples.  The "encoder" pathway is simply discarded.</figcaption>

At test time, when we want to generate new samples, we simply input values of $z\sim\mathcal{N}(0,I)$ into the decoder.
That is, we remove the "encoder", including the multiplication and addition operations that would change the distribution of $z$.
This (remarkably simple) test-time network is shown in Figure 4.

## How is a VAE different compared to a determenistic autoencoder?

To be added

## Is the encoding\decoding with VAEs a lossless process?

To be added

Section 5 in [2]

See second reviwer's question for the BB-ANS paper

https://openreview.net/forum?id=ryE98iR5tm

# What is Bits Back Coding? 

Bits back coding illustrated by an example taken from [3]

<img src="/assets/images/mixture_gaussian.png" alt="" style="width:100%">
<figcaption><strong>Figure 5</strong>: The most natural source model may produce multiple codewords for a given symbol. (a) shows a source with a single binaryhidden variable which identities from which Gaussian,$G_1$ or $G_2$, the symbol value is sampled. Values of $x$ near $x_0$ are likely to have come from either Gaussian. (b) shows the resulting coding density effectively used if we were to always pick the shorter codeword. This density wastes coding space because it is incorrectly shaped and has an area signicantly less than unity.</figcaption>

Consider a source that outputs real numbers that are distributed according to a mixture of two Gaussians.  These numbers are truncated to some precision to form a set of symbols. The component distributions and the output distribution are shown in Figure 5a, where the truncation effect is left out for the sake of graphical simplicity. **The most natural source model to use in this case is one that requires one bit to specify from which Gaussian a givensymbol was produced plus however many bits are needed tocode the symbol using that Gaussian.** However, the identity of the Gaussian that produced a given symbol is often ambiguous, in particular, a number near $x_0$ is likely to have come from either Gaussian. In these cases the source model maps each symbol to two codewords: one for each Gaussian. **If we were to always pick the shorter of the two codewords, we would effectively be assuming that the symbols were distributed as in Figure 1b.  However, this distribution is obviously incorrect (it is not even normalized) and will lead to suboptimal compression.**

**The obvious way around this problem is to use a one-to-one code that is based on the mixture distribution; that is, we assign a codeword to each symbol based on its total probability mass, obtained by summing the contributions from each Gaussian.  We show that the same efficiency can be achieved by picking codewords stochastically from a one-to-many code.** This may seem surprising, since for a given symbol both codewords in the one-to-many source code are longer than the codeword in the one-to-one sourcecode. However, we will show that extra information can becommunicated through the choice of codewords.

<img src="/assets/images/bits_back_model.png" alt="" style="width:100%">
<figcaption> <strong>Figure 6</strong>: A scheme in which auxiliary data is communicated along with the primary symbol data in order to achieve optimal compressionwhen the source code produces multiple codewords for a given symbol. </figcaption>

The bits communicated in the auxiliary data will more than make up for the excess codeword lengths that result from not always using the shortest codeword.This method of encoding is shown in Figure 6.  Notice that if the sender does not wish to communicate an extra stream of auxiliary data to the receiver, some of the primary symbol data can be set aside and used as auxiliary data.  (It may need to be XORed with a pseudo-random bit stream to make it appear more random to the sender.)Suppose in the Gaussian mixture example that a sender wishes to encode a truncated value,$x_0$, that is twice as likely under $G_1$ as it is under $G_2$, and that 2 bits are required to encode the truncated value under $G_1$. Including the single bitrequired to specify which Gaussian is being used, an optimal source code (possibly arithmetic) will thus have codewordswith lengths $l_1=3$ bits and $l_1=4$ bits. If the sender always picks the shorter codeword, the average codeword length is 3 bits. Suppose instead that whenever the sender must communicate the particular symbol $x_0$, the sender chooses between each of the two codewords equally often (in general, the ratio of choices will depend on the truncated value).  It appears that the average codeword length in this case is.$\frac{l_1+l_2}{2}=3.5$ bits, which is higher than that obtained by always choosing the shorter codeword. However, this cost is effectively lowered because the receiver can recover information from the choice of codeword in the following manner. Say the sender has well-compressed auxiliary dataavailable in the form of a queued bit stream with '0' and '1' having equal frequency. When encoding $x_0$, the sender uses the next bit in the auxiliary data queue to choose between $G_1$ and $G_2$. The sender then produces a codeword that will have an average length of 3.5 bits. (It is important to note that this codeword specifies which of $G_1$ and $G_2$ is being used.) When decoding, the receiver reads off the bit that says which Gaussian was used and then determines the truncated value $x_0$ from the codeword. Given the decoded value, the receiver can run the same encoding algorithm as the sender used,and determine that a choice of equal probability was made between $G_1$ and $G_2$ . Since the receiver also knows whichGaussian was selected, the receiver can recover the queued auxiliary data bit that was used to make the choice. In this way, on average 1 bit of the auxiliary data is communicatedat no extra cost. These recovered bits are called "bits-back". If the auxiliary data is useful, the average effective codeword length is reduced by 1 bit due to the savings, giving an effective average length of 2.5 bits - less than the 3 bits required by the shortest codeword. We refer to this method of stochastic source coding as **bits-back** coding. It is important to note that the ratio of choices depends on the symbol being encoded. For example, if the truncated value is far to the right of $x_0$ in Figure 1a, then picking the codewords equally often would be very inefficient, since the codeword under $G_1$ would be extremely long, making the benefit of the single recovered bit negligible. In this case the sender should pick $G_1$ much less often and, as a result, the sender will read offonly `part of a bit' from the auxiliary data queue to determine which codeword to use.


## Why would we choose one symbol to many codeword scheme (BB coding) over a one-to-one coding scheme?

## What is Assymetrical Numeral System?

## How does BB+ANS interfacing work?

## Why is ANS preferred over arithmetic coding?

# References

[1] Auto-Encoding Variational Bayes. https://arxiv.org/pdf/1312.6114.pdf

[2] Tutorial on Variational Autoencoders. https://arxiv.org/pdf/1606.05908.pdf

[3] Efficient Stochastic Source Codingand an Application to a BayesianNetwork Source Model. https://www.cs.toronto.edu/~hinton/absps/freybitsback.pdf

[4] Practical lossless compression with latent variables using bits back coding. https://arxiv.org/pdf/1901.04866.pdf

[5] Bit-Swap: Recursive Bits-Back Coding for Lossless Compression withHierarchical Latent Variables https://arxiv.org/pdf/1905.06845.pdf

[6] Improving Data Compression Based On Deep Learning. https://fhkingma.com/improvingdata/mscthesis.pdf


