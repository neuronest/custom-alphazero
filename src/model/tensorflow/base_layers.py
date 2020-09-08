import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, Activation, BatchNormalization, Add, Conv2D
from tensorflow.keras.regularizers import l2

# https://github.com/tensorflow/tensorflow/issues/32477#issuecomment-556032114
BatchNormalization._USE_V2_BEHAVIOR = False


def policy_loss(y_true, y_pred):
    return tf.reduce_mean(tf.reduce_sum(-y_true * K.log(y_pred + K.epsilon()), axis=-1))


def value_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(tf.subtract(y_pred, y_true)))


class InnerConvBlock(Layer):
    def __init__(
        self,
        input_dim,
        filters,
        kernel_size,
        strides=(1, 1),
        padding="same",
        activation=None,
        batch_normalization=False,
        l2_penalization_term=1e-4,
    ):
        super(InnerConvBlock, self).__init__()
        self.activation = activation
        self.batch_normalization = batch_normalization
        if input_dim is not None:
            self.conv_layer = Conv2D(
                input_shape=input_dim,
                filters=filters,
                kernel_size=kernel_size,
                padding=padding,
                strides=strides,
                kernel_regularizer=l2(l2_penalization_term),
            )
        else:
            self.conv_layer = Conv2D(
                filters,
                kernel_size,
                padding=padding,
                strides=strides,
                kernel_regularizer=l2(l2_penalization_term),
            )
        self.activation_layer = (
            Activation(self.activation) if self.activation is not None else None
        )
        self.batch_normalization_layer = (
            BatchNormalization() if self.batch_normalization else None
        )

    @tf.function
    def call(self, inputs, training=False):
        outputs = inputs
        outputs = self.conv_layer(outputs)
        if self.batch_normalization:
            outputs = self.batch_normalization_layer(outputs)
        if self.activation is not None:
            outputs = self.activation_layer(outputs)
        return outputs


class OuterConvBlock(Layer):
    def __init__(
        self,
        input_dim,
        filters,
        kernel_size,
        padding="same",
        activation=None,
        batch_normalization=False,
        residual_connexion=False,
        l2_penalization_term=1e-4,
    ):
        super(OuterConvBlock, self).__init__()
        self.activation = activation
        self.residual_connexion = residual_connexion
        self.input_dim = input_dim
        self.filters = filters
        self.inner_conv_1 = InnerConvBlock(
            input_dim=input_dim,
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            activation=activation,
            batch_normalization=batch_normalization,
            l2_penalization_term=l2_penalization_term,
        )
        self.inner_conv_2 = InnerConvBlock(
            input_dim=None,
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            activation=None,
            batch_normalization=batch_normalization,
            l2_penalization_term=l2_penalization_term,
        )
        if self.residual_connexion:
            self.residual_connexion = InnerConvBlock(
                input_dim=input_dim,
                filters=filters,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding=padding,
                activation=None,
                batch_normalization=batch_normalization,
                l2_penalization_term=l2_penalization_term,
            )
            self.add_residual = Add()
        else:
            self.residual_connexion, self.add_residual = None, None
        self.activation_layer = Activation(self.activation)

    @tf.function
    def call(self, inputs, training=False):
        outputs = inputs
        outputs = self.inner_conv_1(outputs)
        outputs = self.inner_conv_2(outputs)
        if self.residual_connexion:
            identity_resized = self.residual_connexion(inputs)
            outputs = self.add_residual([identity_resized, outputs])
        outputs = self.activation_layer(outputs)
        return outputs
