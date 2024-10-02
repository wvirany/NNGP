import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.initializers import RandomNormal


# Set up Seaborn plotting style
sns.set_style("darkgrid",
              {"axes.facecolor": ".95",
               "axes.edgecolor": "#000000",
               "grid.color": "#EBEBE7",
               "font.family": "serif",
               "axes.labelcolor": "#000000",
               "xtick.color": "#000000",
               "ytick.color": "#000000",
               "grid.alpha": 0.4 })


class SingleLayerNetwork(keras.Model):
    """
    A single-layer neural network model.
    """

    def __init__(self, num_units, activation='tanh'):
        """
        Initialize the network with specified number of units and activation function.
        """
        super().__init__()

        self.net = Sequential([
            # Hidden layer
            Dense(num_units, input_shape=(1,), activation=activation,
                kernel_initializer=RandomNormal(mean=0.0, stddev=5.0),
                bias_initializer=RandomNormal(mean=0.0, stddev=5.0)),
            # Output layer
            Dense(1, activation='linear',
                kernel_initializer=RandomNormal(mean=0.0, stddev=1/np.sqrt(num_units)),
                bias_initializer=RandomNormal(mean=0.0, stddev=0.1))
        ])

    def call(self, X):
        """
        Forward pass of the network.
        """
        return self.net(X)
    

def nngp_exp1(num_units, inputs, num_samples=1000):
    """
    Run experiment to generate outputs from multiple network instantiations.
    """
    x = []
    y = []

    for i in range(num_samples):
        net = SingleLayerNetwork(num_units=num_units)

        x.append(net(inputs[0]))
        y.append(net(inputs[1]))

    return x, y


# Define input points
inputs = np.array([[-.2], [.4]])

# Run experiments with different numbers of hidden units
x1, y1 = nngp_exp1(1, inputs=inputs, num_samples=1000)
x2, y2 = nngp_exp1(3, inputs=inputs, num_samples=1000)
x3, y3 = nngp_exp1(10, inputs=inputs, num_samples=1000)

outputs = [(x1, y1), (x2, y2), (x3, y3)]

# Set up the plot
fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, sharex=True, figsize=(12, 4))

fig.suptitle('Convergence of priors to Gaussian process for single-input networks')

# Set x and y limits for all subplots
axes[0].set_xlim(-3, 3)
axes[0].set_ylim(-3, 3)

# Plot the results
for i in range(3):
   data = outputs[i]
   axes[i].scatter(data[0], data[1], s=.5, c='midnightblue')
   axes[i].set_xlabel(f'$N$ = {i}')

plt.tight_layout()
plt.show()