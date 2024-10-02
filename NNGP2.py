import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras.losses import MeanSquaredError

import imageio.v2 as imageio
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


np.random.seed(42)


# Set up Seaborn style
sns.set_style({
    "axes.facecolor": "#f7f9fc",
    "axes.edgecolor": "#000000",
    "grid.color": "#EBEBE7",
    "font.family": "serif",
    "axes.labelcolor": "#000000",
    "xtick.color": "#000000",
    "ytick.color": "#000000",
    "grid.alpha": 0.4
})


class SingleLayerNetwork(keras.Model):
    """ A single-layer neural network """

    def __init__(self, num_units, activation=tf.math.sin):
        super().__init__()
        self.net = Sequential([
            Dense(num_units, input_shape=(1,), activation=activation,
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1.0),
                  bias_initializer=RandomNormal(mean=0.0, stddev=1.0)),
            Dense(1, activation='linear',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1/np.sqrt(num_units)),
                  bias_initializer=RandomNormal(mean=0.0, stddev=.1))
        ])
        self.optimizer = Adam(learning_rate=0.0006)
        self.loss_fn = MeanSquaredError()

    def call(self, X):
        return self.net(X)

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            loss = self.loss_fn(y, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss


# Set up the experiment parameters
noise_scale = 1e-1
f = lambda x: 5 * np.sin(x)
train_points = 5


# Generate training data
x_train = np.random.uniform(-np.pi, np.pi, train_points)
y_train = f(x_train) + noise_scale * np.random.normal(0, 1, train_points)
x_test = np.linspace(-np.pi, np.pi, 50)


# Create an ensemble of neural networks
nn_ensemble = [SingleLayerNetwork(num_units=256) for _ in range(50)]


def create_frame(epoch, nn_ensemble, x_train, y_train, x_test, filename):
    """
    Create a single frame of GIF. Trains each network in the ensemble
    for one optimization step. Plots the predictions of each neural network
    and the mean of the ensemble.
    """

    plt.figure(figsize=(10, 6))
    plt.rcParams.update({
        "mathtext.fontset": "cm",
        "font.family": "sans-serif",
        "text.usetex": False,
    })

    preds = []

    for nn in nn_ensemble:
        # Train the network
        loss = nn.train_step(x_train, y_train)
        
        # Get predictions
        pred = nn(x_test)
        preds.append(pred)
        plt.plot(x_test, pred.numpy().flatten(), c='pink', lw=.7, alpha=.5, zorder=1)

    mean = np.mean(preds, axis=0).flatten()
    std = np.std(preds, axis=0).flatten()

    plt.plot(x_test, mean, c='firebrick', lw=1, label='NN Ensemble Mean', zorder=1)
    plt.scatter(x_train, y_train, c='k', s=25, zorder=2)
    plt.fill_between(x_test, mean - 2 * std, mean + 2 * std, color='lightblue', alpha=0.2, zorder=1)

    plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
               ['$-\\pi$', '$-\\pi/2$', '$0$', '$\\pi/2$', '$\\pi$'])
    plt.yticks([-5, 0, 5], ['$-5$', '$0$', '$5$'])
    plt.xlim(-np.pi, np.pi)
    plt.ylim(-7, 7)
    plt.text(2.15, -6.25, f'$t = {epoch}$', {"fontsize":18})

    pink_line = Line2D([], [], color='pink', lw=1, label='NN Ensemble')
    blue_line = Line2D([], [], c='firebrick', lw=1.5, label='Mean Estimator')
    plt.legend(handles=[pink_line, blue_line], loc='upper left')

    plt.savefig(filename)
    plt.close()


# Main training loop and GIF creation
def main():
    frames = []
    for epoch in range(100):
        filename = f"frame_{epoch:03d}.png"
        create_frame(epoch, nn_ensemble, x_train, y_train, x_test, filename)
        frames.append(imageio.imread(filename))
        os.remove(f"frame_{epoch:03d}.png")
        
    # Create GIF
    imageio.mimsave('nngp_animation.gif', frames, fps=25)
        

if __name__ == "__main__":
    main()