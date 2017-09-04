![alt tag](trained_models/VAE_MNIST/posterior_likelihood_evolution.gif)

_Visualization of the 2D latent variable corresponding to a Convolutional Variational Autoencoder during training on MNIST dataset (handwritten digits). The image to the left is the mean of the approximate posterior Q(z|X) and each color represents a class of digits within the dataset. The image to the left shows samples from the decoder (likelihood) P(X|z). The title above shows the iteration number and total loss [Reconstruction + KL] of the model at the point that images below were produced from the model under training. One can observe that by the time the generated outputs (left image) get better, the points on the latent space (posterior) also get into better seperated clusters. Also note that points get closer to each other because the KL part of the total loss is imposing a zero mean gaussian distribution on the latent variable, which is realized on the latent variable as the trainig proceeds._

Tensorflow implementation of various generative models based on CNNs. 

Use these code with no warranty and please respect the accompanying license.

# [**Generative Adversarial Networks**](GenerativeAdversarialNetworks)
### [**Deep Convolutional Generative Adversarial Networks (DCGANs) - Radford et al. 2015**](GenerativeAdversarialNetworks/DCGAN.ipynb)
### [**Image-to-Image Translation with Conditional Adversarial Networks - Isola et al. 2016**](GenerativeAdversarialNetworks/img2imgGAN.ipynb)
### [**Wasserstein GAN - Arjovsky et al. 2017**](GenerativeAdversarialNetworks/WGAN.py)
### [**Improved Triaing of Wasserstein GANs - Gulrajani et al. 2017**](GenerativeAdversarialNetworks/WGAN2.py)

# [**Variational Auto-Encoders**](VariationalAutoEncoders)
### [**Auto-Encoding Variational Bayes - Kingma and Welling 2013**](VariationalAutoEncoders/VAE.ipynb)
<!-- ### [**Semi-Supervised Learning with Deep Generative Models - Kingma et al. 2014**](VariationalAutoEncoders/cVAE.ipynb) -->


