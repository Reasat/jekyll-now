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

**What is VAE?** <br/>
$$ P(x) = \int P(X|z; \theta) P(z)dz $$ <br/>
$$ P(X) $$ is data distribution $$z$$ is the latent variable. <br/> 
$$P(X|z; \theta)$$ is the likelihood function. 

**Why does encoding\decoding with VAEs produce a lossless compression?**<br/>
**Why not use a simple autoencoder?**<br/>
**What is BB? What is ANS?**<br/>
**How does BB+ANS interfacing work. Why ANS preferred over arithmetic coding**<br/>
