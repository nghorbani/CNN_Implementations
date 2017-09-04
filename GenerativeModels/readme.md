# [**Generative Adversarial Networks**](DCGAN.ipynb)

Ian J. Goodfellow et al 2014_Universite de Montreal
### Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
Radford et al 2016_indico Research and Facebook Research

In Adversarial training procedure two models are trained together. The generative model, G, that estimates the data distribution and the discriminative model, D, that determines if a given sample has come from the dataset or artificially generated. G is evolved into making artificially generated samples that are with higher probability mistaken by the D model as coming from true data distribution. One nice property of GANs is that the generator is not directly updated with data examples, but by the gradients coming through the discriminator. Previously Deep Convolutional GANs (Neural Networks) were not that easy to train. The second paper offers some guidelines which makes deep convolutional GANs (DCGAN) easier to train.

Class sweep: With the same z slowly move the class labels to smoothly generate different numbers

![alt tag](../trained_models/DCGAN_MNIST/class_sweep/class_sweep.gif)

Z-Space interpolaton: Vary a coefficient that determines how much of two different Z values are used to sample from the generator.

![alt tag](../trained_models/DCGAN_MNIST/zspace_sweep/zspace_sweep.gif)

Above we see the results after 200 epochs of training with Adam update rule and learning rate of 0.0002 and beta1 of 0.5 on MNIST dataset.
### [Image-to-Image Translation with Conditional Adversarial Networks](img2imgGAN.ipynb)
Isola et al 2016_Berkeley AI Research (BAIR) Laboratory

##### Translation of CMP Images to Labels (A2B)
![alt tag](../trained_models/img2imgGAN_CMP_A2B/generated_3.jpg)

##### Translation of CMP Labels to Images (B2A)
![alt tag](../trained_models/img2imgGAN_CMP_B2A/generated_1.jpg)

# [**Variational Autoencoders**](VAE.ipynb)
### Auto-Encoding Variational Bayes
Kingma and Welling, 2013_Universiteit van Amsterdam

![alt tag](../trained_models/VAE_MNIST/posterior_likelihood_evolution.gif)

Visulization of a Convolutional Variational Auto-Encoder with 2D latent space Z during training on MNIST dataset (handwritten digits). The Image to the right is the mean of the approximate posterior Q(z|X) and each color represents a class of digits within the dataset. Image to the left shows samples from the likelihood P(X|z) as a way to visualize the predictive prior. The title above shows the iteration number and total loss of the model at the point that images below were produced from the model under training. What we observe is the by the time the generated outputs (left image) get better, the points in the posterior also get into better seperated clusters.


