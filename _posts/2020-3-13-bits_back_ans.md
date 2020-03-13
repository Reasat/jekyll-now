---
layout: post
title: "Trying to understand bits-back coding"
mathjax: true
---

**Practical lossless compression with latent variables using bits back coding**<br/>
*James Townsend, Thomas Bird, David Barber*<br/>
https://arxiv.org/pdf/1901.04866.pdf

**Components of the compression pipeline**
**VAE** used to encode input to latent space and also decode input from latent space. This works as the first level of compression?
**BB-ANS** used to further compress the latent space.

**What is a VAE?** <br/>
https://arxiv.org/pdf/1312.6114.pdf
From https://arxiv.org/pdf/1606.05908.pdf
$$X={\{x^i\}}^{N}_{i=1}$$, are the data samples, $$z$$ is the latent variable <br/>
$$ p_{\theta}(x)$$ is the marginal likelihood <br/>
$$p_{\theta}(z)$$ is the prior and $$\int p_{\theta}(x|z) $$ is the posterior likelihood <br/>
$$p_{\theta}(x|z)$$ is the likelihood function. 
 $$\int p_{\theta}(x|z)$$ is computationally intractable, this is approximated by $$\int q_{\theta}(x|z)$$

**Why does encoding\decoding with VAEs produce a lossless compression?**<br/>
**Why not use a simple autoencoder?**<br/>
**What is BB? What is ANS?**<br/>
**How does BB+ANS interfacing work. Why ANS preferred over arithmetic coding**<br/>
