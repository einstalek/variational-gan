import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


def build_face_net(input_shape=(224, 224, 3), std=0.02, slope=0.2):
    """
    Backbone convolutional net for critic
    
    :param input_shape: input image shape
    :param std: std for Normal initializer
    :param slope: slope for LeakyReLU
    """
    inp = layers.Input(shape=input_shape)
    init = RandomNormal(stddev=std)
    X = layers.Conv2D(32, 5, strides=(2, 2), padding='same', kernel_initializer=init)(inp)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU(slope)(X)
    
    X = layers.Conv2D(64, 5, strides=(2, 2), padding='same', kernel_initializer=init)(X)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU(slope)(X)
    
    X = layers.Conv2D(128, 5, strides=(2, 2), padding='same', kernel_initializer=init)(X)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU(slope)(X)
    
    X = layers.Conv2D(256, 5, strides=(2, 2), padding='same', kernel_initializer=init)(X)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU(slope)(X)
    
    X = layers.Conv2D(512, 5, strides=(2, 2), padding='same', kernel_initializer=init)(X)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU(slope)(X)
    
    X = layers.Conv2D(1024, 5, strides=(2, 2), padding='same', kernel_initializer=init)(X)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU(slope)(X)
    X = layers.Flatten()(X)
    return Model(inp, X)


def build_critic(input_shape=(224, 224, 3),
                 rate=0.8):
    """
    Critic net
    
    :param input_shape: input image shape
    :param rate: Dropout rate to apply before output
    """
    face_net = build_face_net(input_shape)

    origin = layers.Input(shape=input_shape)
    distored = layers.Input(shape=input_shape)
    
    latent_origin = face_net(origin)
    latent_distored = face_net(distored)

    X = layers.Lambda(lambda x: K.abs(x[0]-x[1]))([latent_origin, latent_distored])
    X = layers.Dropout(rate)(X)
    X = layers.Dense(1)(X)
    return Model([origin, distored], X)


def build_encoder(input_shape=(224, 224, 3), lat_dim=100, slope=0.2, std=0.02, rate=0.8):
    """
    Encoder net
    
    :param lat_dim: Latent dimension size
    :param slope: slope for LeakyReLU
    :param std: std for Normal initializer
    :param rate: Dropout rate to apply before output
    """
    init = RandomNormal(stddev=std)
    inp = layers.Input(shape=input_shape)

    X = layers.Conv2D(32, 5, strides=(2, 2), padding='same', 
                      kernel_initializer=init)(inp)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU(slope)(X)
    
    X = layers.Conv2D(64, 5, strides=(2, 2), padding='same', 
                      kernel_initializer=init)(X)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU(slope)(X)
    
    X = layers.Conv2D(128, 5, strides=(2, 2), padding='same', 
                      kernel_initializer=init)(X)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU(slope)(X)
    
    X = layers.Conv2D(256, 5, strides=(2, 2), padding='same', 
                      kernel_initializer=init)(X)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU(slope)(X)
    
    X = layers.Conv2D(512, 5, strides=(2, 2), padding='same', 
                      kernel_initializer=init)(X)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU(slope)(X)
    
    X = layers.Conv2D(1024, 5, strides=(2, 2), padding='same', 
                      kernel_initializer=init)(X)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU(slope)(X)
   
    X = layers.Flatten()(X)
    X = layers.Dropout(rate=rate)(X)
    X = layers.Dense(lat_dim)(X)
    return Model(inp, X)


def build_generator(lat_dim=100, std=0.02):
    """
    Generator net 
    
    :param lat_dim: Latent dimension size
    :param std: std for Normal initializer
    """
    init = RandomNormal(stddev=std)
    inp = layers.Input(shape=(lat_dim,))
    
    X = layers.Dense(14*14*1024, kernel_initializer=init)(inp)
    X = layers.Reshape((14, 14, 1024))(X)
    X = layers.BatchNormalization()(X)
    X = layers.ReLU()(X)
    
    X = layers.Conv2DTranspose(512, 5, strides=(2, 2), padding="same", 
                               kernel_initializer=init)(X)
    X = layers.BatchNormalization()(X)
    X = layers.ReLU()(X)
    
    X = layers.Conv2DTranspose(256, 5, strides=(2, 2), padding="same", 
                               kernel_initializer=init)(X)
    X = layers.BatchNormalization()(X)
    X = layers.ReLU()(X)
    
    X = layers.Conv2DTranspose(128, 5, strides=(2, 2), padding="same", 
                               kernel_initializer=init)(X)
    X = layers.BatchNormalization()(X)
    X = layers.ReLU()(X)
    
    X = layers.Conv2DTranspose(3, 5, strides=(2, 2), padding="same", activation="tanh", 
                               kernel_initializer=init)(X)
    return Model(inp, X)


def build_lat_discriminator(lat_dim=100, slope=0.2, std=0.02, rate=0.5):
    """
    Binary classificator in latent space
    
    :param lat_dim: Latent dimension size
    :param slope: slope for LeakyReLU
    :param std: std for Normal initializer
    :param rate: Dropout rate to apply after each layer
    """
    init = RandomNormal(stddev=std)
    inp = layers.Input(shape=(lat_dim,))
    
    X = layers.Dense(1024, kernel_initializer=init)(inp)
    X = layers.LeakyReLU(slope)(X)
    X = layers.Dropout(rate)(X)
    
    X = layers.Dense(1024, kernel_initializer=init)(X)
    X = layers.LeakyReLU(slope)(X)
    X = layers.Dropout(rate)(X)
    
    X = layers.Dense(1024, kernel_initializer=init)(X)
    X = layers.LeakyReLU(slope)(X)
    X = layers.Dropout(rate)(X)

    X = layers.Dense(1)(X)
    return Model(inp, X)


def build_discriminator(input_shape=(224, 224, 3), slope=0.2, std=0.02, rate=0.8):
    """
    Discriminator net
    
    :param slope: slope for LeakyReLU
    :param std: std for Normal initializer
    :param rate: Dropout rate to apply before output
    """
    init = RandomNormal(stddev=std)
    inp = layers.Input(shape=input_shape)

    X = layers.Conv2D(64, 5, strides=(2, 2), padding='same', kernel_initializer=init)(inp)
    X = layers.LeakyReLU(slope)(X)
    
    X = layers.Conv2D(128, 5, strides=(2, 2), padding='same', kernel_initializer=init)(X)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU(slope)(X)
    
    X = layers.Conv2D(256, 5, strides=(2, 2), padding='same', kernel_initializer=init)(X)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU(slope)(X)
    
    X = layers.Conv2D(512, 5, strides=(2, 2), padding='same', kernel_initializer=init)(X)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU(slope)(X)
    
    X = layers.Conv2D(1024, 5, strides=(2, 2), padding='same', kernel_initializer=init)(X)
    X = layers.BatchNormalization()(X)
    X = layers.LeakyReLU(slope)(X)
   
    X = layers.Flatten()(X)
    X = layers.Dropout(rate=rate)(X)
    X = layers.Dense(1)(X)
    return Model(inp, X)



