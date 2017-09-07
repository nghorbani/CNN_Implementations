![alt tag](common/images/vae_posterior_likelihood_evolution.gif)

_Visualization of the 2D latent variable corresponding to a Convolutional Variational Autoencoder during training on MNIST dataset (handwritten digits). The image to the left is the mean of the approximate posterior Q(z|X) and each color represents a class of digits within the dataset. The image to the left shows samples from the decoder (likelihood) P(X|z). The title above shows the iteration number and total loss [Reconstruction + KL] of the model at the point that images below were produced from the model under training. One can observe that by the time the generated outputs (left image) get better, the points on the latent space (posterior) also get into better seperated clusters. Also note that points get closer to each other because the KL part of the total loss is imposing a zero mean gaussian distribution on the latent variable, which is realized on the latent variable as the trainig proceeds._

Tensorflow implementation of various generative models based on convolutional neural networks. Throughout different models i will always keep the same architecture for decoder/encoder/discriminator. This helps comparision of models that only differ by their cost functions. 

Use these code with no warranty and please respect the accompanying license.

# Generative Adversarial Networks
### [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks - Radford et al. 2015](https://arxiv.org/abs/1511.06434)
[Jupyter Notebook](Notebooks/DCGAN.ipynb) 
[Python code](GenerativeModels/DCGAN.py)
### [Image-to-Image Translation with Conditional Adversarial Networks- Isola et al. 2016](https://arxiv.org/abs/1611.07004)
[Jupyter Notebook](Notebooks/img2imgGAN.ipynb)
[Python Code](GenerativeModels/img2imgGAN.py)
### [Wasserstein GAN - Arjovsky et al. 2017](https://arxiv.org/abs/1701.07875)
[Python Code](GenerativeModels/WGAN.py)
### [Improved Training of Wasserstein GANs - Gulrajani et al. 2017](https://arxiv.org/abs/1704.00028)
[Python Code](GenerativeModels/WGAN2.py)

# Variational Autoencoders
### [Auto-Encoding Variational Bayes - Kingma and Welling 2013](https://arxiv.org/abs/1312.6114)
[Jupyter Notebook](Notebooks/VAE.ipynb)
[Python Code](GenerativeModels/VAE.py)

# Hybrid Models
### [Adversarial Autoencoders - Makhzani et al. 2015](https://arxiv.org/abs/1511.05644)
[Jupyter Notebook](Notebooks/AAE.ipynb)
[Python Code](GenerativeModels/AAE.py)

# Basic Models
### Convolutional Denoising Autoencoders
[Jupyter Notebook](Notebooks/CDAE.ipynb)
[Python Code](GenerativeModels/CDAE.py)


