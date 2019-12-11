# -*- coding: utf-8 -*-

import tensorflow as tf

from tensorflow.keras import layers

from utils import sparse_to_tuple


class GraphConvolution(layers.Layer):
    """
    Graph Convolutional Layer as presented by the work of Kipf & Welling:
    "Semi-Supervised Classification with Graph Convolutional Networks".
    Implemented as a TensorFlow/Keras layer (for version of TF 2.0).

    Disclaimer
    ----------
        This is a simplified version for education purposes, for a full implementation of the
        graph neural network I suggest you to visit the official repository at
        https://github.com/tkipf/gcn or the keras implementation at
        https://github.com/tkipf/keras-gcn

    Parameters
    ----------

    units : int
        Number of convolutional filters (i.e. output size of the layer).
    adjacency : scipy sparse matrix
        adjacency matrix of the GCN.
    activation : str or function
        Activation function to use. If you don't specify anything, no activation is
        applied (ie. "linear" activation: a(x) = x).
    use_bias: bool
        Whether the layer uses a bias vector.
    kernel_initializer : str or function
        Initializer for the kernel weights matrix (see initializers).
    bias_initializer :
        Initializer for the bias vector (see initializers).
    kernel_regularizer :
        Regularizer function applied to the kernel weights matrix (see regularizer).
    bias_regularizer :
        Regularizer function applied to the bias vector (see regularizer).
    activity_regularizer :
        Regularizer function applied to the output of the layer (its "activation"). (see regularizer).
    kernel_constraint :
        Constraint function applied to the kernel weights matrix (see constraints).
    bias_constraint :
        Constraint function applied to the bias vector (see constraints).
    """
    def __init__(self,
                 units,
                 adjacency,
                 activation=None,
                 use_bias=False,
                 kernel_initializer="glorot_uniform",
                 bias_initializer="zeros",
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        self.units = units
        self.adjacency = adjacency

        self.activation = tf.keras.activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.bias_initializer = tf.keras.initializers.get(bias_initializer)
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self.activity_regularizer = tf.keras.regularizers.get(activity_regularizer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.bias_constraint = tf.keras.constraints.get(bias_constraint)

    def build(self, input_shape):
        indices, values, shape = sparse_to_tuple(self.adjacency)
        self.A = tf.SparseTensor(
            indices=indices,
            values=values,
            dense_shape=shape
        )

        self.W = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer=self.kernel_initializer,
                                 name="W",
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint,
                                 trainable=True)

        if self.use_bias:
            self.b = self.add_weight(shape=(self.units,),
                                     initializer=self.bias_initializer,
                                     name="b",
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint,
                                     trainable=True)

    def call(self, inputs):
        convolution = tf.sparse.sparse_dense_matmul(self.A, inputs)
        output = tf.matmul(convolution, self.W)

        if self.use_bias:
            output += self.b

        return self.activation(output)

    def compute_output_shape(self, input_shape):
        batch_size = input_shape.get_shape().as_list()[0]
        return batch_size, self.units
