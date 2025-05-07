# Copyright (c) 2020, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class MaskCNN(nn.Module):
    r"""
    Masking Convolutional Neural Network.

    Adds padding to the output of the module based on the given lengths.
    This is to ensure that the results of the model do not change when batch
    sizes change during inference. Input needs to be in the shape of
    (batch_size, channel, hidden_dim, seq_len).

    Inspired by https://github.com/SeanNaren/deepspeech.pytorch/blob/master/model.py
    Copyright (c) 2017 Sean Naren, MIT License.

    Args:
        sequential (nn.Sequential): A sequential container of convolutional layers.

    Inputs: inputs, seq_lengths
        - **inputs** (torch.Tensor): Input tensor of shape BxCxHxT
        - **seq_lengths** (torch.Tensor): The actual length of each sequence
          in the batch (Tensor of shape B).

    Returns: output, seq_lengths
        - **output** (torch.Tensor): Masked output tensor from the sequential layers.
        - **seq_lengths** (torch.Tensor): Sequence lengths of the output tensor.
    """

    def __init__(self, sequential: nn.Sequential) -> None:
        super(MaskCNN, self).__init__()
        self.sequential = sequential

    def _get_sequence_lengths(self, module: nn.Module, seq_lengths: Tensor) -> Tensor:
        r"""
        Calculate the output sequence lengths after a convolutional or pooling layer.

        Args:
            module (nn.Module): The convolutional or pooling layer.
            seq_lengths (torch.Tensor): Input sequence lengths (shape B).

        Returns:
            torch.Tensor: Output sequence lengths (shape B).
        """
        if isinstance(module, (nn.Conv2d, nn.MaxPool2d)):
            # Formula for Conv2d: floor((L_in + 2*padding - dilation*(kernel_size - 1) - 1) / stride) + 1
            # Assumes calculation is done on the 'Time' dimension (dim 3, index 1 for kernel/padding etc.)

            if isinstance(module, nn.Conv2d):
                kernel_size = module.kernel_size[1]
                stride = module.stride[1]
                padding = module.padding[1]
                dilation = module.dilation[1]
            elif isinstance(module, nn.MaxPool2d):
                # Note: This uses MaxPool2d parameters. If kernel_size/stride/padding/dilation
                # are not tuples, they apply to both H and W dimensions.
                # We assume the relevant parameters for the sequence length (T dimension)
                # are accessible via index 1 if they are tuples.
                kernel_size = module.kernel_size if isinstance(
                    module.kernel_size, int) else module.kernel_size[1]
                stride = module.stride if isinstance(
                    module.stride, int) else module.stride[1]
                padding = module.padding if isinstance(
                    module.padding, int) else module.padding[1]
                dilation = module.dilation if isinstance(
                    module.dilation, int) else module.dilation[1]

                # --- Simplified calculation used in original code for MaxPool2d ---
                # The original code assumed MaxPool2d always halves the sequence length,
                # which is true for kernel_size=2, stride=2.
                # Keep the original logic for MaxPool2d if it's specifically intended.
                # If more general pooling is used, the formula below is better.
                if kernel_size == 2 and stride == 2 and padding == 0 and dilation == 1:
                    # Faster equivalent of floor(L_in / 2)
                    return seq_lengths >> 1

            # General formula calculation (works for both Conv2d and general MaxPool2d)
            numerator = seq_lengths + 2 * padding - \
                dilation * (kernel_size - 1) - 1
            new_lengths = torch.div(
                numerator.float(), stride, rounding_mode='floor') + 1
            return new_lengths.int()

        # Return unchanged if the module doesn't affect sequence length
        return seq_lengths

    def forward(self, inputs: Tensor, seq_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass applies layers and masks the output."""
        output = inputs  # Start with the initial input

        for module in self.sequential:
            output = module(output)
            current_seq_lengths = self._get_sequence_lengths(
                module, seq_lengths)

            # Create mask directly on the correct device
            mask = torch.zeros_like(output, dtype=torch.bool)

            # Mask padded regions
            for i, length in enumerate(current_seq_lengths):
                actual_length = length.item()
                # output shape is BxCxHxT, mask needs to cover dimension T (index 3)
                if output.size(3) > actual_length:
                    mask[i, :, :, actual_length:].fill_(True)

            output = output.masked_fill(mask, 0.0)
            seq_lengths = current_seq_lengths  # Update lengths for the next layer
            # The output of this layer becomes the input for the next
            # No need to reassign 'inputs = output' as we modify 'output' in place conceptually

        return output, seq_lengths


class VGGExtractor(nn.Module):
    r"""
    VGG extractor for automatic speech recognition.

    Described in "Advances in Joint CTC-Attention based End-to-End Speech
    Recognition with a Deep CNN Encoder and RNN-LM"
    (https://arxiv.org/pdf/1706.02737.pdf). Applies 2 VGG blocks (Conv-Conv-Pool).

    Args:
        input_dim (int): Dimension of the input feature (e.g., 80 for Mel spectro.).
                         This corresponds to the 'H' dimension in BxCxHxT.
        in_channels (int): Number of input channels (default: 1).
        out_channels (Tuple[int, int]): Number of output channels for
                                         the two VGG blocks (default: (64, 128)).

    Inputs: inputs, input_lengths
        - **inputs** (torch.Tensor): Input tensor of shape (batch, time, input_dim).
        - **input_lengths** (torch.Tensor): Tensor of sequence lengths of shape (batch).

    Returns: outputs, output_lengths
        - **outputs** (torch.Tensor): Output tensor of shape
          (batch, time_out, channels_out * dim_out).
        - **output_lengths** (torch.Tensor): Tensor of output sequence lengths
          of shape (batch).
    """

    def __init__(
        self,
        input_dim: int,
        in_channels: int = 1,
        out_channels: Tuple[int, int] = (64, 128),
    ) -> None:
        super(VGGExtractor, self).__init__()
        assert len(
            out_channels) == 2, "out_channels should be a tuple of two integers"
        self.input_dim = input_dim
        self.in_channels = in_channels
        self.out_channels = out_channels

        vgg_block1_out = out_channels[0]
        vgg_block2_out = out_channels[1]

        sequential = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, vgg_block1_out, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=vgg_block1_out),
            nn.ReLU(),
            nn.Conv2d(vgg_block1_out, vgg_block1_out, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=vgg_block1_out),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Halves H and T dimensions

            # Block 2
            nn.Conv2d(vgg_block1_out, vgg_block2_out, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=vgg_block2_out),
            nn.ReLU(),
            nn.Conv2d(vgg_block2_out, vgg_block2_out, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=vgg_block2_out),
            nn.ReLU(),
            # Halves H and T dimensions again
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv = MaskCNN(sequential)

        # Calculate the output dimension H after two MaxPool2d(2, stride=2)
        # Each pooling layer halves the dimension (integer division)
        self._output_height = self.input_dim // 2 // 2
        self._output_feature_dim = self.out_channels[1] * self._output_height

    def get_output_dim(self) -> int:
        """Returns the flattened feature dimension per time step."""
        return self._output_feature_dim

    def forward(self, inputs: Tensor, input_lengths: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Forward pass through the VGG extractor.

        Args:
            inputs (torch.Tensor): shape (batch, time, input_dim)
            input_lengths (torch.Tensor): shape (batch)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - outputs: shape (batch, time_out, channels_out * height_out)
                - output_lengths: shape (batch)
        """
        # Reshape input to BxCxHxT for Conv2d
        # Input: (batch, time, dim) -> (batch, 1, time, dim)
        outputs = inputs.unsqueeze(1)
        # Transpose to: (batch, 1, dim, time) - Matches MaskCNN's BxCxHxT expectation
        outputs = outputs.transpose(2, 3)

        # Apply MaskCNN (handles convolutions, pooling, masking, length calculation)
        # output shape from MaskCNN: (batch, out_channels[1], H_out, T_out)
        # H_out = input_dim // 4
        # T_out is calculated based on input_lengths
        outputs, output_lengths = self.conv(outputs, input_lengths)

        # Reshape output for downstream layers (e.g., RNN)
        # (batch, C_out, H_out, T_out) -> (batch, T_out, C_out, H_out)
        batch_size, _, _, time_out = outputs.size()
        outputs = outputs.permute(0, 3, 1, 2)

        # Flatten C_out and H_out dimensions
        # (batch, T_out, C_out, H_out) -> (batch, T_out, C_out * H_out)
        # Use the pre-calculated output dimension
        outputs = outputs.contiguous().view(
            batch_size, time_out, self._output_feature_dim)

        return outputs, output_lengths
