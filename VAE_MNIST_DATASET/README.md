
### MNIST Image Reconstruction using Variational Autoencoder (VAE)

This project demonstrates image reconstruction and generation using a Variational Autoencoder (VAE) trained on the MNIST handwritten digit dataset.

The model learns a compressed latent representation of handwritten digits and reconstructs the original images from this latent space. Unlike traditional autoencoders, the VAE learns a probabilistic latent distribution, allowing it to generate new digit samples by sampling random latent vectors.

#### Features
- Train a VAE on MNIST digits
- Reconstruct handwritten digit images
- Generate new digit samples from latent space
- Visualize the 2D latent representation
- GPU/CPU support with PyTorch

#### Architecture
- Encoder: `784 -> 256 -> latent mean/log variance`
- Latent dimension: `20`
- Decoder: `20 -> 256 -> 784`

#### Loss Function
The VAE optimizes:
- Reconstruction Loss (Binary Cross Entropy)
- KL Divergence Regularization

#### Libraries
- Python
- PyTorch
- Torchvision
- Matplotlib

#### Results
The trained model can:
- Reconstruct noisy handwritten digits
- Learn meaningful latent representations
- Generate realistic MNIST types samples



```bash
 make clean
 make 
 source  torch_env/bin/activate
```
