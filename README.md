# Code for Variational GAN with critic

This project was developed as a part of my Master's research. 
The main idea is to use critic network to improve GAN's ability to reconstruct original pictures

## Model architecture 
Final Model is composed of:
- Encoder (E)
- Generator (G)
- Discriminator (D)
- Latent space discriminator (C)
- Critic-network (S)

<img src="./model-scheme.jpg " width="340" height="340">

## Usage
#### Data preparation
Download [celebA](https://www.kaggle.com/jessicali9530/celeba-dataset) dataset
#### Training a model
Put path to dataset directory in train.py in CriticGan instance and run: 
```
python train.py
```

## Results
#### Sampling from latent space 
* Faces, randomly generated from latent space

<img src="./random-faces.png " width="400" height="400">

####  Reconstruction from latent space

Faces from original dataset          |  Reconstructions
:-------------------------:|:-------------------------:
<img src="./reconstruction-real.png " width="400" height="400">  |  <img src="./reconstruction-fake.png " width="400" height="400">

## Links
* [Paper on variational GAN with critic](https://docs.google.com/document/d/1BN6-4jeCU4xXMLtFPaltRlhMjnVWl82j1AcAVYUm5PE/edit?usp=sharing) (in Russian)

* [Model's weights pretrained on celebA dataset](https://drive.google.com/file/d/1nR4MkdTwpixklpKZNX45t-5ZRf_GTAaz/view?usp=sharing)
