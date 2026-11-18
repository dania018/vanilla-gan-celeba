# Vanilla GAN: Generating Faces with PyTorch

A **Generative Adversarial Network (GAN)** built from scratch in PyTorch that learns to generate brand-new human faces from random noise, trained on the **CelebA** dataset of 202,599 celebrity photos.

> None of these faces are real. The generator invents each one from 100 random numbers.

## What is a GAN?

A GAN is made of **two neural networks competing against each other**:

| Network | Input | Output | Goal |
|---|---|---|---|
| **Generator** | 100 random numbers (noise) | A fake 64×64 RGB face | Fool the discriminator |
| **Discriminator** | A face image (real or fake) | Probability it's real | Catch the fakes |

The generator keeps improving at creating realistic faces, and the discriminator keeps improving at spotting fakes. This adversarial game pushes the generator to produce more and more convincing images.

### Preprocessing

```python
transforms.Compose([
    transforms.CenterCrop(178),              # 178×218 → 178×178 (square face crop)
    transforms.Resize(64),                   # → 64×64
    transforms.ToTensor(),                   # pixels → [0, 1]
    transforms.Normalize((0.5,)*3, (0.5,)*3) # → [-1, 1], matching the generator's Tanh output
])
```

A custom `ImageProcessor` dataset class loads every image from the folder, applies the transforms, and feeds a `DataLoader` with **batch size 128**, shuffled.

## Model architecture

Both networks use **fully connected (dense) layers**, which is what makes this a *vanilla* GAN.

### Generator

```
Noise (100)
 → Linear(100, 256)  + ReLU
 → Linear(256, 512)  + ReLU
 → Linear(512, 1024) + ReLU
 → Linear(1024, 12288) + Tanh     # 12288 = 64 × 64 × 3
 → reshape → image (3 × 64 × 64)
```

### Discriminator

```
Image (3 × 64 × 64)
 → Flatten (12288)
 → Linear(12288, 1024) + LeakyReLU(0.2)
 → Linear(1024, 512)   + LeakyReLU(0.2)
 → Linear(512, 256)    + LeakyReLU(0.2)
 → Linear(256, 1)      + Sigmoid  # probability of "real"
```

**Why these activations?**
- **Tanh** in the generator's output keeps pixels in [−1, 1], matching the normalised real images.
- **LeakyReLU** in the discriminator lets a small gradient through for negative values, so the generator still gets useful feedback.
- **Sigmoid** in the discriminator's output gives a real/fake probability for binary cross-entropy.

## Training

| Setting | Value |
|---|---|
| Loss | `BCELoss` (binary cross-entropy) |
| Optimizer | Adam for both networks, `lr = 0.0002`, `betas = (0.5, 0.999)` |
| Batch size | 128 |
| Device | Apple Silicon (MPS), with CUDA/CPU fallback |

Each batch runs two steps:

**1. Train the discriminator**
- Real images are labelled **1** and generated images **0**
- `d_loss = (BCE(D(real), 1) + BCE(D(fake), 0)) / 2`
- Fake images are `.detach()`-ed, so this step updates only the discriminator

**2. Train the generator**
- Generate fakes and label them **1**, the "real" label
- `g_loss = BCE(D(fake), 1)`, which rewards the generator when the discriminator is fooled

After every epoch, the model saves a grid of 8 generated faces to show how it's progressing.

## Results

| Epoch 1 | Epoch 12 |
|---|---|
| ![Epoch 1](assets/epoch_01.png) | ![Epoch 12](assets/epoch_12.png) |

- Rough face shapes, skin tones and hair appear within the **first epoch**
- Losses stay balanced through training (**G-loss ≈ 1.0–1.5, D-loss ≈ 0.55–0.65**), so neither network overpowers the other
- Later epochs show more varied features such as hairstyles, backgrounds and even sunglasses
- The images are still **noisy**, which is expected from a dense-layer GAN that doesn't model spatial structure

## Tech stack

Python · PyTorch · Torchvision · NumPy · Matplotlib · Pillow
