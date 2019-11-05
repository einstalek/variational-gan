import tensorflow as tf
from models import *
from utils import pic, DataGenerator

from IPython import display
import matplotlib.pyplot as plt

entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


class CriticGan:
    def __init__(self,
                 dirp,
                 image_shape,
                 lat_dim,
                 batch_size=64):
        self.dirp = dirp
        self.encoder = build_encoder(input_shape=image_shape,
                                     lat_dim=lat_dim)
        self.gen = build_generator(lat_dim=lat_dim)
        self.lat_discr = build_lat_discriminator(lat_dim=lat_dim)
        self.discr = build_discriminator(input_shape=image_shape)
        self.critic = build_critic(input_shape=image_shape)

        self.lat_dim = lat_dim
        l_rate, b1 = 0.0001, 0.6
        # Optimizers
        self.enc_opt = tf.keras.optimizers.Adam(l_rate, b1)
        self.lat_discr_opt = tf.keras.optimizers.Adam(l_rate, b1)
        self.gen_opt = tf.keras.optimizers.Adam(l_rate, b1)
        self.discr_opt = tf.keras.optimizers.Adam(l_rate, b1)
        self.critic_opt = tf.keras.optimizers.Adam(l_rate, b1)
        self.clip_value = 10.
        self.batch_size = batch_size
        self.data = DataGenerator(batch_size)

    @tf.function
    def train_step(self, real_images, distored_images):
        """
        One trainig step
        :param real_images: batch of real_images
        :param distored_images: batch of distorted

        """
        with tf.GradientTape() as gen_tape, \
                tf.GradientTape() as discr_tape, \
                tf.GradientTape() as enc_tape, \
                tf.GradientTape() as ldiscr_tape, \
                tf.GradientTape() as critic_tape:
            encoded = self.encoder(real_images, training=True)
            noise = tf.random.normal(shape=[self.batch_size, self.lat_dim])
            reconstructed = self.gen(encoded, training=True)
            fake_images = self.gen(noise, training=True)
            noise_codes_output = self.lat_discr(noise, training=True)
            real_codes_output = self.lat_discr(encoded, training=True)
            fake_output = self.discr(fake_images, training=True)
            real_output = self.discr(real_images, training=True)
            reconstructed_output = self.discr(reconstructed, training=True)
            critic_reconstructed = self.critic([real_images, reconstructed], training=False)
            critic_distored = self.critic([real_images, distored_images], training=False)

            critic_loss = entropy(0.95 * tf.ones_like(critic_distored), critic_distored) + \
                          entropy(tf.zeros_like(critic_reconstructed), critic_reconstructed)

            reconstructed_loss = entropy(tf.ones_like(critic_reconstructed), critic_reconstructed)

            enc_loss = reconstructed_loss + entropy(tf.ones_like(real_codes_output), real_codes_output)

            gen_loss = reconstructed_loss + entropy(tf.ones_like(fake_output), fake_output) + \
                       entropy(tf.ones_like(reconstructed_output), reconstructed_output)

            discr_loss = entropy(0.95 * tf.ones_like(real_output), real_output) + \
                         entropy(tf.zeros_like(fake_output), fake_output) + \
                         entropy(tf.zeros_like(reconstructed_output), reconstructed_output)

            lat_discr_loss = entropy(0.95 * tf.ones_like(noise_codes_output), noise_codes_output) + \
                             entropy(tf.zeros_like(real_codes_output), real_codes_output)

            gen_grads = gen_tape.gradient(gen_loss, self.gen.trainable_variables)
            gen_grads = [tf.clip_by_norm(grad, self.clip_value) for grad in gen_grads]
            lat_discr_grads = ldiscr_tape.gradient(lat_discr_loss, self.lat_discr.trainable_variables)
            enc_grads = enc_tape.gradient(enc_loss, self.encoder.trainable_variables)
            enc_grads = [tf.clip_by_norm(grad, self.clip_value) for grad in enc_grads]
            discr_grads = discr_tape.gradient(discr_loss, self.discr.trainable_variables)
            critic_grads = critic_tape.gradient(critic_loss, self.critic.trainable_variables)

        self.gen_opt.apply_gradients(zip(gen_grads, self.gen.trainable_variables))
        self.lat_discr_opt.apply_gradients(zip(lat_discr_grads, self.lat_discr.trainable_variables))
        self.enc_opt.apply_gradients(zip(enc_grads, self.encoder.trainable_variables))
        self.discr_opt.apply_gradients(zip(discr_grads, self.discr.trainable_variables))
        self.critic_opt.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
        return gen_loss, lat_discr_loss, enc_loss, discr_loss

    def generate_images(self, gen, encoder, n, shape=(224, 224, 3)):
        """
        Display intermediate results of generator
        """
        x = pic(self.dirp, n).reshape((1, *shape))
        real_code = encoder.predict(x)
        code = tf.random.normal(shape=[1, self.lat_dim])
        _x = gen.predict(code)
        _img = gen(code)
        plt.figure(figsize=(12, 4))
        plt.subplot(121)
        plt.imshow(_img[0, :, :])
        plt.axis('off')
        plt.subplot(122)
        _img = gen(real_code)
        plt.imshow(_img[0, :, :])
        plt.axis('off')
        plt.show()

    def train(self, epochs=30):
        for epoch in range(1, epochs + 1):
            for i, (origin, distored) in enumerate(self.data):
                gl, ldl, el, dl = self.train_step(origin, distored)
                if i % 10 == 0:
                    display.clear_output(wait=True)
                    self.generate_images(self.gen, self.encoder, 11)
                    print("E-%d I-%d" % (epoch, (i + 1)))
                    print("Generator loss: %s" % str(gl.numpy()))
                    print("Encoder loss: %s" % str(el.numpy()))
                    print("Discriminator loss: %s" % str(dl.numpy()))


if __name__ == "__main__":
    gan = CriticGan("path_to_images_dir",
                    image_shape=(224, 224, 3),
                    lat_dim=100)
    gan.train(100)