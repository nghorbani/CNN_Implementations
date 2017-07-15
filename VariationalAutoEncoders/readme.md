# [**Convolutional Variational Encoders**](VAE_MNIST.ipynb)
### Auto-Encoding Variational Bayes
Kingma and Welling, 2013_Universiteit van Amsterdam

![alt tag](trained_models/VAE_MNIST/posterior_likelihood_evolution.gif)

Visulization of a Convolutional Variational Auto-Encoder with 2D latent space Z during training on MNIST dataset (handwritten digits). The Image to the right is the mean of the approximate posterior Q(z|X) and each color represents a class of digits within the dataset. Image to the left shows samples from the likelihood P(X|z) as a way to visualize the predictive prior. The title above shows the iteration number and total loss of the model at the point that images below were produced from the model under training. What we observe is the by the time the generated outputs (left image) get better, the points in the posterior also get into better seperated clusters.
