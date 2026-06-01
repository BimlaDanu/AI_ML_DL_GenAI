#### Image Reconstruction using CNN (MNIST & CIFAR-10)

This project demonstrates image reconstruction using Convolutional Neural Networks (CNNs) and Autoencoders


The model learns to:
1. Compress images into a smaller latent representation
2. Reconstruct the original image from noisy or compressed inputs

The CNN encoder extracts important visual features using convolution filters, while the decoder reconstructs the image using transposed convolutions.

This project helps understand:
- CNN feature extraction
- Image denoising and reconstruction
- Latent space representations
- Basics of generative deep learning models

The implementation includes:
- Data preprocessing and normalization
- CNN-based autoencoder architecture
- Training and reconstruction visualization
- Loss vs epoch plots
- Learned filter visualization

The notebook [CNN_classification.ipynb](CNN_Pytorch/CCNN_classification.ipynb) trained on CPU for small epoch sizes and is not efficient. For efficient training  see the notebook [CNN_Classifier_Pytorch.ipynb](CNN_Pytorch/CNN_Classifier_Pytorch.ipynb) which created and trained on Google Colab with GPU. It might show rendering problem. In that case for the details of code please have a look on files [CNN_Classifier_Pytorch_CIFAR.py](CNN_Classifier_Pytorch_CIFAR.py) and [CNN_Pytorch/CCNN_Classifier_Pytorch_MNIST.py](CNN_Pytorch/CNN_Classifier_Pytorch_MNIST.py).




#### Bash commands and python environments

```bash
 make clean
 make 
 source  torch_env/bin/activate
```

#### Image reconstruction and classification using CNN

See for the details: [Torchvision documentation](https://docs.pytorch.org/vision/main/index.html)

#### Compatible with python 3.11.3 Environment

```python
 !pip uninstall torch torchvision -y
 !pip install torch==2.3.1 torchvision==0.18.1
 !pip show torch torchvision
```


#### CUDA Environment

```python
 !pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

 import torch
 import torchvision
 !pip show torch torchvision

 print(torch.__version__)
 print(torchvision.__version__)
```
