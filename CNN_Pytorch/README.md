## Convolutional Neural Network(CNN) training on  MNIST & CIFAR-10

The two CNN projects built in PyTorch — a **denoising autoencoder** on MNIST and an **image classifier** on CIFAR-10.

---

### Table of Contents

1. [Project 1 — MNIST Denoising Autoencoder](#project-1--mnist-denoising-autoencoder)
2. [Project 2 — CIFAR-10 CNN Classifier](#project-2--cifar-10-cnn-classifier)
3. [Mathematical Details](#mathematics-details)
4. [Summary](#summary)
5. [Requirements](#requirements)

---

The notebook[CNN_Classifier_Pytorch.ipynb](CNN_Classifier_Pytorch.ipynb) is created and trained on Google Colab with GPU. It might show rendering problem. In that case for the details of code please have a look on files [CNN_Classifier_Pytorch_CIFAR.py](CNN_Classifier_Pytorch_CIFAR.py) and [CNN_Classifier_Pytorch_MNIST.py](CNN_Classifier_Pytorch_MNIST.py).

### Project 1 — MNIST Denoising Autoencoder

#### Overview

A convolutional autoencoder trained to reconstruct clean handwritten digit images from artificially corrupted/noisy inputs. The model learns a compressed latent representation that captures the essential structure of each digit, discarding noise in the process.

```
Input (noisy)  →  Encoder  →  Latent Space  →  Decoder  →  Output 
  1×28×28           ↓            64×7×7           ↓          1×28×28
                32×14×14                        32×14×14
```

#### Architecture

| Layer | Type | Input Shape | Output Shape | Parameters |
|---|---|---|---|---|
| enc_conv1 | Conv2d + BN + LeakyReLU | 1×28×28 | 32×14×14 | 3×3 kernel, stride 2 |
| enc_conv2 | Conv2d + BN + LeakyReLU | 32×14×14 | 64×7×7 | 3×3 kernel, stride 2 |
| dec_conv1 | ConvTranspose2d + BN + ReLU | 64×7×7 | 32×14×14 | 3×3 kernel, stride 2 |
| dec_conv2 | ConvTranspose2d + Sigmoid | 32×14×14 | 1×28×28 | 3×3 kernel, stride 2 |

#### Mathematics

#### 1. Noise Injection

Gaussian noise is added to each input image before training:

$$\tilde{x} = x + \epsilon, \quad \epsilon \sim \mathcal{N}(0,\, \sigma^2), \quad \sigma = 0.2$$

The result is clipped to remain a valid image:

$$\tilde{x} = \text{clip}(\tilde{x},\; 0,\; 1)$$

#### 2. Convolution Operation

Each convolutional layer applies a learned filter $W$ over the input feature map $X$:

$$Z^{(l)}_{i,j} = \sum_{m}\sum_{p,q} W^{(l)}_{m,p,q} \cdot X^{(l-1)}_{m,\, i \cdot s + p,\, j \cdot s + q} + b^{(l)}$$

where $s$ is the stride and $(p, q)$ index the kernel spatial dimensions.

**Output spatial size** after a conv with stride $s$, padding $p$, kernel $k$:

$$H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1$$

For the first encoder layer: $\lfloor (28 + 2 - 3) / 2 \rfloor + 1 = 14$

#### 3. Transposed Convolution (Decoder Upsampling)

The decoder uses transposed convolutions to upsample back to the original resolution:

$$H_{out} = (H_{in} - 1) \cdot s - 2p + k + p_{out}$$

where $p_{out}$ is `output_padding` (set to 1 to recover exact dimensions).

#### 4. Batch Normalisation

Applied after each convolution to stabilise training:

$$\hat{z} = \frac{z - \mu_\mathcal{B}}{\sqrt{\sigma^2_\mathcal{B} + \epsilon}}, \qquad y = \gamma \hat{z} + \beta$$

where $\mu_\mathcal{B}$ and $\sigma^2_\mathcal{B}$ are the mini-batch mean and variance, and $\gamma$, $\beta$ are learnable scale and shift parameters.

#### 5. Leaky ReLU (Encoder Activation Function)

Prevents dying neurons by allowing a small gradient for negative inputs:

$$\text{LeakyReLU}(z) = \begin{cases} z & z > 0 \\ \alpha z & z \leq 0 \end{cases}, \quad \alpha = 0.2$$

#### 6. Sigmoid Output Activation

Squashes decoder output to $[0, 1]$ to match normalised pixel values:

$$\text{Sigmoid}(z) = \frac{1}{1 + e^{-z}}$$

#### 7. Loss Function — Mean Squared Error

The model minimises pixel-wise reconstruction error between output and clean target:

$$\mathcal{L}_{MSE} = \frac{1}{N} \sum_{i=1}^{N} \left\| f(\tilde{x}_i) - x_i \right\|^2$$

where $f(\cdot)$ is the full encoder–decoder mapping.

#### 8. Optimiser — Adam

Parameters are updated using adaptive moment estimation:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

with $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\eta = 10^{-3}$.

#### 9. Learning Rate Schedule — StepLR

The learning rate decays by factor $\gamma$ every $T$ epochs:

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t / T \rfloor}, \quad \gamma = 0.5,\; T = 15$$

#### 10. Mixed Precision Training (AMP)

Forward pass and loss computed in **FP16**; gradients scaled to prevent underflow:

$$g_{\text{scaled}} = g \cdot s, \qquad \theta \mathrel{-}= \frac{\eta \cdot g_{\text{scaled}}}{s}$$

The scaler $s$ is dynamically adjusted each step.

---

### Project 2 — CIFAR-10 CNN Classifier

#### Overview

A three-block convolutional neural network trained to classify 32×32 colour images into one of 10 categories: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

```
Input          → Block 1 → Block 2 → Block 3 → FC Head → Output
3×32×32          32×16×16  64×8×8    128×4×4    512→256    10 classes
```

#### Architecture

| Layer | Type | Output Shape | Notes |
|---|---|---|---|
| Block 1 | Conv2d + BN + ReLU + MaxPool | 32×16×16 | 32 filters, 3×3 |
| Block 2 | Conv2d + BN + ReLU + MaxPool | 64×8×8 | 64 filters, 3×3 |
| Block 3 | Conv2d + BN + ReLU + MaxPool | 128×4×4 | 128 filters, 3×3 |
| FC 1 | Linear + ReLU + Dropout(0.4) | 512 | Flatten: 128×4×4 = 2048 |
| FC 2 | Linear + ReLU + Dropout(0.3) | 256 | — |
| FC 3 | Linear | 10 | Raw logits |

#### Mathematics

#### 1. Data Augmentation

During training only, two stochastic transforms are applied:

**Random Horizontal Flip** — each image is flipped with probability 0.5:

$$x' = \text{flip}(x) \quad \text{with } P = 0.5$$

**Random Crop** — a $32 \times 32$ crop is extracted from a zero-padded $40 \times 40$ image:

$$x' = \text{crop}(x_{\text{padded}},\; 32 \times 32)$$

#### 2. Normalisation

Input pixels are mapped from $[0, 1]$ to $[-1, 1]$ per channel:

$$x' = \frac{x - \mu}{\sigma}, \quad \mu = (0.5, 0.5, 0.5),\; \sigma = (0.5, 0.5, 0.5)$$

To display images, the inverse transform is applied:

$$x = x' \cdot \sigma + \mu$$

#### 3. Max Pooling

Reduces spatial resolution by a factor of 2 (stride = kernel = 2), selecting the maximum activation per region:

$$Z^{(l)}_{i,j} = \max_{p,q \in \{0,1\}} X^{(l)}_{2i+p,\; 2j+q}$$

#### 4. Fully Connected Layer

Each neuron computes a weighted sum of all inputs:

$$z = W \mathbf{h} + b, \quad W \in \mathbb{R}^{d_{out} \times d_{in}}$$

The number of input features after flattening Block 3 output:

$$d_{in} = 128 \times 4 \times 4 = 2048$$

#### 5. Dropout Regularisation

During training, each neuron is zeroed independently with probability $p$, and the remaining activations are rescaled:

$$h'_i = \begin{cases} 0 & \text{with probability } p \\ \dfrac{h_i}{1 - p} & \text{with probability } 1 - p \end{cases}$$

Applied with $p = 0.4$ after FC1 and $p = 0.3$ after FC2.

#### 6. Loss Function — Cross-Entropy

For a $C$-class classification problem, the loss over a mini-batch of size $N$ is:

$$\mathcal{L}_{CE} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{z_{y_i}}}{\sum_{c=1}^{C} e^{z_c}}$$

where $z_c$ is the raw logit for class $c$ and $y_i$ is the true label. This is equivalent to applying **softmax** then **negative log-likelihood**:

$$\hat{p}_c = \text{softmax}(z)_c = \frac{e^{z_c}}{\sum_{c'} e^{z_{c'}}}, \qquad \mathcal{L}_{CE} = -\frac{1}{N}\sum_i \log \hat{p}_{y_i}$$

#### 7. L2 Regularisation (Weight Decay)

An L2 penalty is added to the loss to discourage large weights:

$$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \sum_\theta \theta^2, \quad \lambda = 10^{-4}$$

In Adam, this is implemented via `weight_decay=1e-4`.

#### 8. Learning Rate Schedule — ReduceLROnPlateau

The learning rate is halved when validation loss has not improved for 5 consecutive epochs:

$$\eta \leftarrow \eta \cdot 0.5 \quad \text{if } \mathcal{L}_{val}^{(t)} \geq \min_{t' < t} \mathcal{L}_{val}^{(t')} \text{ for 5 epochs}$$

#### 9. Accuracy Metrics

**Overall accuracy:**

$$\text{Acc} = \frac{\sum_{c=1}^{C} \text{TP}_c}{N_{total}}$$

**Per-class accuracy:**

$$\text{Acc}_c = \frac{\text{TP}_c}{N_c}$$

where $\text{TP}_c$ is the number of correctly predicted samples for class $c$ and $N_c$ is the total number of samples in class $c$.

---

### Mathematical Details 

Both projects share the following building blocks:

#### Backpropagation — Chain Rule

Gradients flow backward through each operation via the chain rule:

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{\partial \mathcal{L}}{\partial Z^{(l)}} \cdot \frac{\partial Z^{(l)}}{\partial W^{(l)}}$$

For a conv layer with output $Z = W * X + b$:

$$\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial Z} * X, \qquad \frac{\partial \mathcal{L}}{\partial X} = \frac{\partial \mathcal{L}}{\partial Z} *_{full} W$$

#### Gradient Scaling (for a Mixed Precision)

To avoid FP16 underflow during backward pass, loss is scaled before `backward()`:

$$\mathcal{L}_{scaled} = s \cdot \mathcal{L}, \qquad g_{true} = \frac{g_{scaled}}{s}$$

The scaler $s$ doubles every 2000 steps and halves on overflow (NaN/Inf detection).

---

### Summary

| Project | Dataset | Epochs | Key Technique | Expected Result |
|---|---|---|---|---|
| Denoising Autoencoder | MNIST (60k train) | 50 | MSE loss, ConvTranspose2d | Loss < 0.02, clean reconstructions |
| CNN Classifier | CIFAR-10 (50k train / 10k test) | 50 | CrossEntropy + Dropout + BN | ~78–82% test accuracy |

---

### Requirements

```
torch >= 2.0
torchvision >= 0.15
matplotlib
```

```bash
pip install torch torchvision matplotlib
```

Both scripts auto-download their datasets on first run via `torchvision.datasets` and will use a CUDA GPU automatically if available.

