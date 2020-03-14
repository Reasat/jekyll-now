---
layout: post
title: "Trying to understand bits-back coding"
mathjax: true
---

**Practical lossless compression with latent variables using bits back coding**<br/>
*James Townsend, Thomas Bird, David Barber*<br/>
https://arxiv.org/pdf/1901.04866.pdf

**Components of the compression pipeline**<br/>
**VAE** used to encode the input to a latent space. This works as the first level of compression?
**BB-ANS** used to further compress the latent space.

**What is a VAE?** <br/>
https://arxiv.org/pdf/1312.6114.pdf

https://arxiv.org/pdf/1606.05908.pdf
<img src="/assets/images/vae_model.png" alt="">

$$X={\{x^i\}}^{N}_{i=1}$$, is the set of data samples, $$z$$ is the latent variable <br/>
$$ p_{\theta}(x)$$ is the marginal likelihood <br/>
$$p_{\theta}(z)$$ is the prior and $$ p_{\theta}(z|x) $$ is the posterior likelihood <br/>
$$p_{\theta}(x|z)$$ is the likelihood function. 
 $$ p_{\theta}(x|z)$$ is computationally intractable, this is approximated by $$ q_{\phi}(x|z)$$

**Why does encoding\decoding with VAEs produce a lossless compression?**<br/>
**Why not use a simple autoencoder?**<br/>
**What is BB? What is ANS?**<br/>
**How does BB+ANS interfacing work. Why ANS preferred over arithmetic coding**<br/>
