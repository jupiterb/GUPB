import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from functools import reduce

from gupb.controller.batman.utils.resources import (
    PATH_TO_REPLY_BUFFER,
    PATH_TO_AUTOENCODER,
)

from stable_baselines3.common.save_util import load_from_pkl
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import gymnasium as gym


class FCNetwork(nn.Module):
    def __init__(self, in_size: int, layer_sizes: list[int]) -> None:
        super(FCNetwork, self).__init__()
        self._fc = nn.Sequential()
        for size in layer_sizes:
            self._fc.append(nn.Linear(in_size, size))
            self._fc.append(nn.ReLU())
            in_size = size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._fc(x)


class ConvBolock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        max_pooling: int | None = None,
    ) -> None:
        super(ConvBolock, self).__init__()
        self._conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=1), nn.ReLU()
        )
        if max_pooling is not None:
            self._conv.append(nn.MaxPool2d(kernel_size=max_pooling, stride=max_pooling))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._conv(x)


class TransposedConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        upsample: int | None = None,
    ) -> None:
        super(TransposedConvBlock, self).__init__()
        self._conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, padding=1),
            nn.ReLU(),
        )
        if upsample is not None:
            self._conv.append(nn.Upsample(scale_factor=upsample, mode="nearest"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._conv(x)


class ConvAutoEncoder(nn.Module):
    def __init__(
        self,
        in_shape: tuple[int],
        channels: list[int],
        kernel_sizes: list[int],
        max_poolings: list[None | int],
        fc_layer_sizes: list[int],
    ) -> None:
        super(ConvAutoEncoder, self).__init__()

        self._latent_size = fc_layer_sizes[-1]

        # Encoder
        self._encoder = nn.Sequential()
        in_channels = in_shape[0]
        for out_channels, kernel_size, max_pooling in zip(
            channels, kernel_sizes, max_poolings
        ):
            self._encoder.append(
                ConvBolock(in_channels, out_channels, kernel_size, max_pooling)
            )
            in_channels = out_channels

        with torch.no_grad():
            conv_out = self._encoder(torch.ones((1, *in_shape))).shape[1:]
            n_flatten = reduce((lambda x, y: x * y), conv_out)

        self._encoder.append(nn.Flatten())
        self._encoder.append(FCNetwork(n_flatten, fc_layer_sizes))

        # Decoder
        self._decoder = nn.Sequential()
        decoder_fc_layer_sizes = list(reversed(fc_layer_sizes[:-1])) + [n_flatten]
        self._decoder.append(FCNetwork(fc_layer_sizes[-1], decoder_fc_layer_sizes))
        self._decoder.append(nn.Unflatten(1, conv_out))

        decoder_channels = list(reversed(channels[:-1])) + [in_shape[0]]
        for out_channels, kernel_size, upsample in zip(
            decoder_channels, reversed(kernel_sizes), reversed(max_poolings)
        ):
            self._decoder.append(
                TransposedConvBlock(in_channels, out_channels, kernel_size, upsample)
            )
            in_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self._encoder(x)
        return self._decoder(latent)

    @property
    def latent_size(self) -> int:
        return self._latent_size

    @property
    def encoder(self) -> nn.Module:
        return self._encoder


def train_autoencoder_with_reply_buffer(
    autoencoder: ConvAutoEncoder,
    num_epochs=10,
    num_reply_buffer_exampples=1000,
    batch_size=64,
    learning_rate=0.001,
) -> ConvAutoEncoder:
    reply_buffer: ReplayBuffer = load_from_pkl(PATH_TO_REPLY_BUFFER)
    observations = reply_buffer.sample(num_reply_buffer_exampples).observations

    # Convert numpy array to PyTorch tensor
    data = torch.tensor(observations, dtype=torch.float32).clone().detach()

    # Create a DataLoader for your dataset
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs in dataloader:
            inputs = inputs[0][:, :, :-1, :-1]
            optimizer.zero_grad()
            outputs = autoencoder(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(dataloader)}"
        )

    print("Training complete")
    return autoencoder


def create_autoencoder(in_shape: tuple[int], latent_size: int) -> ConvAutoEncoder:
    autoencoder = ConvAutoEncoder(
        in_shape, [1, 8, 8], [3, 3, 3], [None, 2, 2], [128, latent_size]
    )
    try:
        autoencoder.load_state_dict(torch.load(PATH_TO_AUTOENCODER))
    except:
        autoencoder = train_autoencoder_with_reply_buffer(
            autoencoder, num_epochs=10, num_reply_buffer_exampples=100
        )
        torch.save(autoencoder.state_dict(), PATH_TO_AUTOENCODER)
    return autoencoder


class EncoderFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim=64) -> None:
        super().__init__(observation_space, features_dim)
        ae = create_autoencoder(observation_space.sample().shape, features_dim)
        self._encoder = ae.encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._encoder(x)
