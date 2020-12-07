# Code for Variational GAN with critic

This project was developed as a part of my Master's research. 
The main idea is to use Сritic network to improve GAN's ability to reconstruct original pictures. Instead of simply using L1 loss to evaluate reconstructed objects, Critic approximates joint distribution on objects pairs. 

## Model architecture 
Final Model is composed of:
- Encoder (E)
- Generator (G)
- Discriminator (D)
- Latent space discriminator (C)
- Critic-network (S)

<img src="./images/model-scheme.jpg " width="340" height="340">

## Usage
#### Data preparation
Download [celebA](https://www.kaggle.com/jessicali9530/celeba-dataset) dataset
#### Training a model
Put path to dataset directory in train.py in CriticGan instance and run: 
```
python train.py
```

## Results
#### Random sampling 
Randomly generated faces          |  Interpolation in latent space
:-------------------------:|:-------------------------:
<img src="./images/random-faces.png " width="400" height="400">  |  <img src="./images/interpolation.png " width="400" height="400">

#### Reconstruction

Faces from original dataset          |  Reconstructions
:-------------------------:|:-------------------------:
<img src="./images/reconstruction-real.png " width="400" height="400">  |  <img src="./images/reconstruction-fake.png " width="400" height="400">

## Links
* [Paper abstract on variational GAN with critic, in English](https://docs.google.com/document/d/1nLn8K2qPqiQy0baZWrhOWw9l6Qq5qBrot4_uuzM7y4g/edit?usp=sharing)

* [Paper on variational GAN with critic, in Russian](https://docs.google.com/document/d/1BN6-4jeCU4xXMLtFPaltRlhMjnVWl82j1AcAVYUm5PE/edit?usp=sharing)

* [Model's weights pretrained on celebA dataset](https://drive.google.com/file/d/1nR4MkdTwpixklpKZNX45t-5ZRf_GTAaz/view?usp=sharing)

