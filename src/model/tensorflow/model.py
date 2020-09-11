import os
import json
import hashlib
import inspect

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.regularizers import l2

from src.config import ConfigModel, ConfigPath
from src.model.tensorflow.base_layers import InnerConvBlock, OuterConvBlock
from src.model.tensorflow.base_layers import policy_loss, value_loss


class ResidualTower(Layer):
    def __init__(
        self,
        input_dim,
        filters,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same",
        activation="relu",
        batch_normalization=True,
        residual_connexion=True,
        depth=20,
        l2_penalization_term=1e-4,
    ):
        super(ResidualTower, self).__init__()
        self.conv_blocks = []
        self.conv_blocks.append(
            InnerConvBlock(
                input_dim=input_dim,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                padding=padding,
                activation=activation,
                batch_normalization=batch_normalization,
                l2_penalization_term=l2_penalization_term,
            )
        )
        for depth_i in range(depth):
            self.conv_blocks.append(
                OuterConvBlock(
                    input_dim=None,
                    filters=filters,
                    kernel_size=kernel_size,
                    padding=padding,
                    activation=activation,
                    batch_normalization=batch_normalization,
                    residual_connexion=residual_connexion,
                    l2_penalization_term=l2_penalization_term,
                )
            )

    @tf.function
    def call(self, inputs, training=False):
        outputs = inputs
        for conv_block in self.conv_blocks:
            outputs = conv_block(outputs)
        return outputs


class PolicyHead(Layer):
    def __init__(
        self,
        output_dim,
        filters=2,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        activation="relu",
        batch_normalization=True,
        l2_penalization_term=1e-4,
    ):
        super(PolicyHead, self).__init__()
        self.inner_conv = InnerConvBlock(
            input_dim=None,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
            batch_normalization=batch_normalization,
            l2_penalization_term=l2_penalization_term,
        )
        self.flatten = Flatten()
        self.dense = Dense(
            units=output_dim,
            activation="softmax",
            kernel_regularizer=l2(l2_penalization_term),
            name="policy_outputs",
        )

    @tf.function
    def call(self, inputs, training=False):
        outputs = inputs
        outputs = self.inner_conv(outputs)
        outputs = self.flatten(outputs)
        outputs = self.dense(outputs)
        return outputs


class ValueHead(Layer):
    def __init__(
        self,
        output_dim,
        hidden_dim=256,
        filters=1,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding="same",
        activation="relu",
        final_activation="tanh",
        batch_normalization=True,
        l2_penalization_term=1e-4,
    ):
        super(ValueHead, self).__init__()
        self.inner_conv = InnerConvBlock(
            input_dim=None,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            activation=activation,
            batch_normalization=batch_normalization,
            l2_penalization_term=l2_penalization_term,
        )
        self.flatten = Flatten()
        self.dense_1 = Dense(
            units=hidden_dim,
            activation=activation,
            kernel_regularizer=l2(l2_penalization_term),
        )
        self.dense_2 = Dense(
            units=output_dim,
            activation=final_activation,
            kernel_regularizer=l2(l2_penalization_term),
            name="value_outputs",
        )

    @tf.function
    def call(self, inputs, training=False):
        outputs = inputs
        outputs = self.inner_conv(outputs)
        outputs = self.flatten(outputs)
        outputs = self.dense_1(outputs)
        outputs = self.dense_2(outputs)
        return outputs


class PolicyValueModel(Model):
    def __init__(
        self,
        input_dim,
        action_space,
        l2_penalization_term=ConfigModel.l2_penalization_term,  # is shared by all layers
        filters=ConfigModel.filters,  # is shared by all layers
        maximum_learning_rate=ConfigModel.maximum_learning_rate,
        momentum=ConfigModel.momentum,
        strides=ConfigModel.strides,  # is shared by all layers
        padding=ConfigModel.padding,  # is shared by all layers
        activation=ConfigModel.activation,  # is shared by all layers
        batch_normalization=ConfigModel.batch_normalization,  # is shared by all layers
        residual_residual_connexion=ConfigModel.residual_residual_connexion,
        residual_depth=ConfigModel.residual_depth,
        residual_kernel_size=ConfigModel.residual_kernel_size,
        policy_kernel_size=ConfigModel.policy_kernel_size,
        value_kernel_size=ConfigModel.value_kernel_size,
        value_hidden_dim=ConfigModel.value_hidden_dim,
    ):
        self.input_dim = input_dim
        self.action_space = action_space
        self.l2_penalization_term = l2_penalization_term
        self.filters = filters
        self.maximum_learning_rate = maximum_learning_rate
        self.momentum = momentum
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.batch_normalization = batch_normalization
        self.residual_residual_connexion = residual_residual_connexion
        self.residual_depth = residual_depth
        self.residual_kernel_size = residual_kernel_size
        self.policy_kernel_size = policy_kernel_size
        self.value_kernel_size = value_kernel_size
        self.value_hidden_dim = value_hidden_dim

        super(PolicyValueModel, self).__init__()
        self.residual_tower = ResidualTower(
            input_dim=self.input_dim,
            filters=self.filters,
            kernel_size=(self.residual_kernel_size, self.residual_kernel_size),
            strides=self.strides,
            padding=self.padding,
            activation=self.activation,
            batch_normalization=self.batch_normalization,
            residual_connexion=self.residual_residual_connexion,
            depth=self.residual_depth,
            l2_penalization_term=self.l2_penalization_term,
        )
        self.policy_head = PolicyHead(
            output_dim=self.action_space,
            filters=self.filters,
            kernel_size=(self.policy_kernel_size, self.policy_kernel_size),
            strides=self.strides,
            padding=self.padding,
            activation=self.activation,
            batch_normalization=self.batch_normalization,
            l2_penalization_term=self.l2_penalization_term,
        )
        self.value_head = ValueHead(
            output_dim=1,
            hidden_dim=self.value_hidden_dim,
            filters=self.filters,
            kernel_size=(self.value_kernel_size, self.value_kernel_size),
            strides=self.strides,
            padding=self.padding,
            activation=self.activation,
            final_activation="tanh",
            batch_normalization=self.batch_normalization,
            l2_penalization_term=self.l2_penalization_term,
        )
        self.optimizer = SGD(
            learning_rate=self.maximum_learning_rate, momentum=self.momentum
        )
        self.compile(optimizer=self.optimizer, loss=[policy_loss, value_loss])
        self(
            np.random.rand(1, *self.input_dim).astype("float32")
        )  # run a dummy forward to initialize the model
        self.steps = 0

    @property
    def hash(self) -> int:
        return sum(
            int(hashlib.md5(str(weight).encode("utf-8")).hexdigest(), 16)
            for weight in self.get_weights()
        )

    def is_equal(self, other: "PolicyValueModel"):
        return self.hash == other.hash

    @tf.function
    def call(self, inputs, training=False, mask=None):
        outputs = inputs
        outputs = self.residual_tower(outputs)
        policy_outputs = self.policy_head(outputs)
        value_outputs = self.value_head(outputs)
        return policy_outputs, value_outputs

    def _load_weights_and_meta(
        self, path, model_prefix, model_meta, model_success=ConfigPath.model_success
    ):
        assert os.path.exists(
            os.path.join(path, model_success)
        ), f"No verification file of the model found at {path}!"
        self.load_weights(os.path.join(path, model_prefix))
        with open(os.path.join(path, model_meta), "r") as fp:
            metadata = json.load(fp)
        self.steps = int(metadata.get("steps"))
        self.update_learning_rate(float(metadata.get("learning_rate")))
        assert self.hash == metadata.get(
            "hash"
        ), "Unexpected weights hash recovered during model loading at {path}!"

    @staticmethod
    def load_model_from_path(
        path, model_prefix=ConfigPath.model_prefix, model_meta=ConfigPath.model_meta
    ):
        with open(os.path.join(path, model_meta), "r") as fp:
            metadata = json.load(fp)
        model = PolicyValueModel(**metadata.get("init_model"))
        model._load_weights_and_meta(path, model_prefix, model_meta)
        return model

    def save_with_meta(
        self,
        path,
        model_prefix=ConfigPath.model_prefix,
        model_meta=ConfigPath.model_meta,
        model_success=ConfigPath.model_success,
    ):
        self.save_weights(os.path.join(path, model_prefix))
        metadata = {
            "steps": int(self.steps),
            "learning_rate": self.get_learning_rate(),
            "hash": self.hash,
            "init_model": dict(
                [
                    (arg, getattr(self, arg))
                    for arg in inspect.signature(self.__init__).parameters
                ]
            ),
        }
        with open(os.path.join(path, model_meta), "w") as fp:
            json.dump(metadata, fp, sort_keys=True, indent=4)
        open(os.path.join(path, model_success), "wb").close()

    def get_learning_rate(self) -> float:
        return float(self.optimizer.learning_rate.numpy())

    def update_learning_rate(self, learning_rate: float):
        self.optimizer.learning_rate.assign(learning_rate)

    def update_momentum(self, momentum: float) -> None:
        self.optimizer.momentum.assign(momentum)
