# Code for Variational-gan

Combining variational and adversarial concepts for generating faces. The main idea is to use critic network to improve GAN's ability to reconstruct original pictures

## Model architecture 
Final Model is composed of:
- Encoder (E)
- Generator (G)
- Discriminator (D)
- Latent space discriminator (C)
- Critic-network (S)

<img src="./model-scheme.jpg " width="340" height="340">

## Results

* Faces, randomly generated from latent space

<img src="./random-faces.png " width="500" height="500">

* Faces from original dataset and thier reconstructions

<img src="./reconstruction-real.png " width="500" height="500">
<img src="./reconstruction-fake.png " width="500" height="500">


## Links
Pretrained weights are available at:
https://drive.google.com/file/d/1nR4MkdTwpixklpKZNX45t-5ZRf_GTAaz/view?usp=sharing
