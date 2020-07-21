import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from src.model.pytorch.base_layers import InnerConvBlock, OuterConvBlock


class ResidualTower(nn.Module):
    def __init__(
        self,
        input_dim,
        filters=256,
        kernel_size=3,
        strides=1,
        batch_normalization=True,
        residual_connexion=True,
        depth=20,
    ):
        super(ResidualTower, self).__init__()
        self.conv_blocks = []
        self.conv_blocks.append(
            InnerConvBlock(
                input_dim=input_dim,
                filters=filters,
                kernel_size=kernel_size,
                strides=strides,
                relu_activation=True,
                batch_normalization=batch_normalization,
            )
        )
        for depth_i in range(depth):
            self.conv_blocks.append(
                OuterConvBlock(
                    input_dim=filters,
                    filters=filters,
                    kernel_size=kernel_size,
                    batch_normalization=batch_normalization,
                    residual_connexion=residual_connexion,
                )
            )

    def forward(self, inputs):
        outputs = inputs
        for conv_block in self.conv_blocks:
            outputs = conv_block(outputs)
        return outputs


class PolicyHead(nn.Module):
    def __init__(
        self,
        input_dim,
        filters,
        output_dim,
        kernel_size=1,
        strides=1,
        batch_normalization=True,
    ):
        super(PolicyHead, self).__init__()
        self.inner_conv = InnerConvBlock(
            input_dim=input_dim[0],
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            relu_activation=True,
            batch_normalization=batch_normalization,
        )
        input_dim_dense = self._forward_conv(torch.rand((1,) + input_dim)).shape[-1]
        self.dense = nn.Linear(in_features=input_dim_dense, out_features=output_dim)

    def _forward_conv(self, inputs):
        outputs = inputs
        outputs = self.inner_conv(outputs)
        flat_outputs = outputs.reshape(outputs.shape[0], -1)
        return flat_outputs

    def forward(self, inputs):
        outputs = inputs
        outputs = self._forward_conv(outputs)
        outputs = F.log_softmax(self.dense(outputs))
        return outputs


class ValueHead(nn.Module):
    def __init__(
        self,
        input_dim,
        filters,
        hidden_dim,
        output_dim,
        kernel_size=1,
        strides=1,
        batch_normalization=True,
    ):
        super(ValueHead, self).__init__()
        self.inner_conv = InnerConvBlock(
            input_dim=input_dim[0],
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            relu_activation=True,
            batch_normalization=batch_normalization,
        )
        input_dim_dense = self._forward_conv(torch.rand((1,) + input_dim)).shape[-1]
        self.dense_1 = nn.Linear(in_features=input_dim_dense, out_features=hidden_dim)
        self.dense_2 = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def _forward_conv(self, inputs):
        outputs = inputs
        outputs = self.inner_conv(outputs)
        flat_outputs = outputs.reshape(outputs.shape[0], -1)
        return flat_outputs

    def forward(self, inputs):
        outputs = inputs
        outputs = self._forward_conv(outputs)
        outputs = self.dense_1(outputs).clamp(min=0)
        outputs = F.tanh(self.dense_2(outputs))
        return outputs


class PolicyValueModel(nn.Module):
    # TODO: learning_rate policy?
    def __init__(
        self,
        input_dim,
        action_space,
        base_filters=256,
        policy_filters=2,
        value_filters=1,
        value_hidden_dim=256,
        output_dim=1,
        depth=20,
        learning_rate=1e-2,
        momentum=0.9,
        l2_penalisation=1e-4,
    ):
        super(PolicyValueModel, self).__init__()
        self.image_height, self.image_width, self.input_channels = input_dim
        self.l2_penalisation = l2_penalisation
        self.residual_tower = ResidualTower(
            self.input_channels, filters=base_filters, depth=depth
        )
        core_outputs_dim = self.residual_tower(
            torch.rand((1, self.input_channels, self.image_height, self.image_width))
        ).shape[1:]
        self.policy_head = PolicyHead(
            input_dim=core_outputs_dim, filters=policy_filters, output_dim=action_space
        )
        self.value_head = ValueHead(
            input_dim=core_outputs_dim,
            filters=value_filters,
            hidden_dim=value_hidden_dim,
            output_dim=output_dim,
        )
        self.optimizer = optim.SGD(
            self.parameters(), lr=learning_rate, momentum=momentum
        )

    def _penalization_term(self, c):
        reg = 0
        for param in self.parameters():
            reg += (param ** 2).sum()
        return c * reg

    def forward(self, inputs):
        outputs = torch.from_numpy(inputs).permute(0, 3, 1, 2).float()
        outputs = self.residual_tower(outputs)
        policy_outputs = self.policy_head(outputs)
        value_outputs = self.value_head(outputs)
        return policy_outputs, value_outputs

    def train_on_batch(self, inputs, labels):
        policy_label, value_label = labels
        policy_label, value_label = (
            torch.tensor(policy_label),
            torch.tensor(value_label),
        )
        self.optimizer.zero_grad()
        policy_outputs, value_outputs = self.forward(inputs)
        value_loss = torch.mean(torch.sum((value_label - value_outputs) ** 2, 1))
        policy_loss = torch.mean(-torch.sum(policy_label * policy_outputs))
        loss_on_batch = (
            value_loss + policy_loss + self._penalization_term(self.l2_penalisation)
        )
        loss_on_batch.backward()
        self.optimizer.step()
        return [loss_on_batch.item()]

    def predict(self, *args):
        policy_outputs, value_outputs = self.__call__(*args)
        return policy_outputs.detach(), value_outputs.detach()
