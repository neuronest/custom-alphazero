import torch.nn as nn


class InnerConvBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        filters,
        kernel_size,
        strides=1,
        relu_activation=True,
        batch_normalization=True,
    ):
        super(InnerConvBlock, self).__init__()
        self.relu_activation = relu_activation
        self.batch_normalization = batch_normalization
        self.conv_layer = nn.Conv2d(
            in_channels=input_dim,
            out_channels=filters,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            stride=strides,
        )
        self.batch_normalization_layer = (
            nn.BatchNorm2d(filters) if self.batch_normalization else None
        )

    def forward(self, inputs):
        outputs = inputs
        outputs = self.conv_layer(outputs)
        if self.batch_normalization:
            outputs = self.batch_normalization_layer(outputs)
        if self.relu_activation:
            outputs = outputs.clamp(min=0)
        return outputs


class OuterConvBlock(nn.Module):
    def __init__(
        self,
        input_dim,
        filters,
        kernel_size,
        batch_normalization=False,
        residual_connexion=None,
    ):
        super(OuterConvBlock, self).__init__()
        self.residual_connexion = residual_connexion
        self.input_dim = input_dim
        self.filters = filters
        self.inner_conv_1 = InnerConvBlock(
            filters=filters,
            kernel_size=kernel_size,
            input_dim=input_dim,
            relu_activation=True,
            batch_normalization=batch_normalization,
        )
        self.inner_conv_2 = InnerConvBlock(
            filters=filters,
            kernel_size=kernel_size,
            input_dim=filters,
            relu_activation=False,
            batch_normalization=batch_normalization,
        )
        if self.residual_connexion:
            self.residual_connexion = InnerConvBlock(
                filters=filters,
                kernel_size=1,
                strides=1,
                input_dim=input_dim,
                relu_activation=False,
                batch_normalization=batch_normalization,
            )
        else:
            self.residual_connexion, self.add_residual = None, None

    def forward(self, inputs):
        outputs = inputs
        outputs = self.inner_conv_1(outputs)
        outputs = self.inner_conv_2(outputs)
        if self.residual_connexion:
            identity_resized = self.residual_connexion(inputs)
            outputs = identity_resized + outputs
        outputs = outputs.clamp(min=0)
        return outputs
