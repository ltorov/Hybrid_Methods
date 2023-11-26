import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from IPython import display
import sys, subprocess

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import idx2numpy
from tensorflow.keras import Sequential
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision, Recall

from src.auxiliary import partitioning
from src.plotting import show_images, show_performance_curve, plot_reconstruction

class GAN(tf.keras.Model):
    """ a basic GAN class 
    Extends:
        tf.keras.Model
    """

    def __init__(self, **kwargs):
        super(GAN, self).__init__()
        self.__dict__.update(kwargs)

        self.gen = tf.keras.Sequential(self.gen)
        self.disc = tf.keras.Sequential(self.disc)

    def generate(self, z):
        return self.gen(z)

    def discriminate(self, x):
        return self.disc(x)

    def compute_loss(self, x):
        """ passes through the network and computes loss
        """
        # generating noise from a uniform distribution
        z_samp = tf.random.normal([x.shape[0], 1, 1, self.n_Z])

        # run noise through generator
        x_gen = self.generate(z_samp)
        # discriminate x and x_gen
        logits_x = self.discriminate(x)
        logits_x_gen = self.discriminate(x_gen)
        ### losses
        # losses of real with label "1"
        disc_real_loss = gan_loss(logits=logits_x, is_real=True)
        # losses of fake with label "0"
        disc_fake_loss = gan_loss(logits=logits_x_gen, is_real=False)
        disc_loss = disc_fake_loss + disc_real_loss

        # losses of fake with label "1"
        gen_loss = gan_loss(logits=logits_x_gen, is_real=True)

        return disc_loss, gen_loss

    def compute_gradients(self, x):
        """ passes through the network and computes loss
        """
        ### pass through network
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            disc_loss, gen_loss = self.compute_loss(x)

        # compute gradients
        gen_gradients = gen_tape.gradient(gen_loss, self.gen.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, self.disc.trainable_variables)

        return gen_gradients, disc_gradients

    def apply_gradients(self, gen_gradients, disc_gradients):

        self.gen_optimizer.apply_gradients(
            zip(gen_gradients, self.gen.trainable_variables)
        )
        self.disc_optimizer.apply_gradients(
            zip(disc_gradients, self.disc.trainable_variables)
        )
    @tf.function
    def train(self, train_x):
        gen_gradients, disc_gradients = self.compute_gradients(train_x)
        self.apply_gradients(gen_gradients, disc_gradients)
        
        
def gan_loss(logits, is_real=True):
    """Computes standard gan loss between logits and labels
    """
    if is_real:
        labels = tf.ones_like(logits)
    else:
        labels = tf.zeros_like(logits)

    return tf.compat.v1.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=logits
    )

def Gan(plotting = True):
    # Initialize parameters 
    TRAIN_BUF=60000
    BATCH_SIZE=512
    TEST_BUF=10000
    DIMS = (28,28,1)
    N_TRAIN_BATCHES =int(TRAIN_BUF/BATCH_SIZE)
    N_TEST_BATCHES = int(TEST_BUF/BATCH_SIZE)
    
    # Import data
    train_images = idx2numpy.convert_from_file('data/train_images.idx3-ubyte')
    train_labels = idx2numpy.convert_from_file('data/train_labels.idx1-ubyte')
    test_images = idx2numpy.convert_from_file('data/test_images.idx3-ubyte')
    test_labels = idx2numpy.convert_from_file('data/test_labels.idx1-ubyte')

    # Split dataset
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype(
        "float32"
    ) / 255.0
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype("float32") / 255.0

    # Batch datasets
    train_dataset = (
        tf.data.Dataset.from_tensor_slices(train_images)
        .shuffle(TRAIN_BUF)
        .batch(BATCH_SIZE)
    )
    test_dataset = (
        tf.data.Dataset.from_tensor_slices(test_images)
        .shuffle(TEST_BUF)
        .batch(BATCH_SIZE)
    )

    # Initialize model parameters
    N_Z = 64
    generator = [
        tf.keras.layers.Dense(units=7 * 7 * 64, activation="relu"),
        tf.keras.layers.Reshape(target_shape=(7, 7, 64)),
        tf.keras.layers.Conv2DTranspose(
            filters=64, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
        ),
        tf.keras.layers.Conv2DTranspose(
            filters=32, kernel_size=3, strides=(2, 2), padding="SAME", activation="relu"
        ),
        tf.keras.layers.Conv2DTranspose(
            filters=1, kernel_size=3, strides=(1, 1), padding="SAME", activation="sigmoid"
        ),
    ]
    discriminator = [
        tf.keras.layers.InputLayer(input_shape=DIMS),
        tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=(2, 2), activation="relu"
        ),
        tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=(2, 2), activation="relu"
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=1, activation=None),
    ]

    # Use optimizers
    gen_optimizer = tf.keras.optimizers.legacy.Adam(0.001, beta_1=0.5)
    disc_optimizer = tf.keras.optimizers.legacy.RMSprop(0.005)# train the model

    # Create model
    model = GAN(
        gen = generator,
        disc = discriminator,
        gen_optimizer = gen_optimizer,
        disc_optimizer = disc_optimizer,
        n_Z = N_Z
    )
    
    # Create pandas dataframe to save the loss information to
    losses = pd.DataFrame(columns = ['disc_loss', 'gen_loss'])

    # Train model
    n_epochs = 1
    for epoch in range(n_epochs):
        # train
        for batch, train_x in tqdm(
            zip(range(N_TRAIN_BATCHES), train_dataset), total=N_TRAIN_BATCHES
        ):
            model.train(train_x)
            
        # test on holdout
        loss = []
        for batch, test_x in tqdm(
            zip(range(N_TEST_BATCHES), test_dataset), total=N_TEST_BATCHES
        ):
            loss.append(model.compute_loss(train_x))
        losses.loc[len(losses)] = np.mean(loss, axis=0)
        # plot results
        display.clear_output()
        print(
            "Epoch: {} | disc_loss: {} | gen_loss: {}".format(
                epoch, losses.disc_loss.values[-1], losses.gen_loss.values[-1]
            )
        )
        if plotting: plot_reconstruction(model)

    if plotting:
        n_epochs = 102
        plt.plot(np.arange(n_epochs),losses['disc_loss'], linestyle = "--", color = 'black', label='Discriminator')
        plt.plot(np.arange(n_epochs),losses['gen_loss'], color = 'black', label='Generator')
        plt.title("Losses")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend()
        plt.show()