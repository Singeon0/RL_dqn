import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np


class ResNetFeatureExtractor(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.resnet = torchvision.models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()

    def forward(self, x):
        return self.resnet(x)


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.feature_extractor = ResNetFeatureExtractor(2)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        """

        :param x: the input tensor containing both mask (channel 0) and ground truth (channel 1) data in shape [batch_size, channels=2, height=110, width=110] ; mask and ground truth are binary images [0, 1]
        :return: the Q value of the input state-action pair in shape [batch_size, 1]
        """
        x = x.float()
        x = self.feature_extractor(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env
        # Assuming env.observation_space.shape is (H, W, C)
        input_channels = self.env.observation_space.shape[
                             2] - 1  # Last dimension is channels AND -1 because i only use mask and image and not the ground truth
        self.feature_extractor = ResNetFeatureExtractor(input_channels)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(self.env.action_space.shape))

        # Batch normalization layers for each channel
        self.bn_image = nn.BatchNorm2d(1)  # BatchNorm for the image channel
        self.bn_mask = nn.BatchNorm2d(1)  # BatchNorm for the mask channel

        # Action rescaling buffers
        self.register_buffer("action_scale",
            torch.tensor(((self.env.action_space.high - self.env.action_space.low) / 2.0).reshape(-1), dtype=torch.float32))
        self.register_buffer("action_bias",
            torch.tensor(((self.env.action_space.high + self.env.action_space.low) / 2.0).reshape(-1), dtype=torch.float32))

    def forward(self, x):
        """
        :param x:  the input tensor containing both image (channel 0) and mask (channel 1) data in shape [batch_size, channels=2, height=110, width=110] ; image is grayscale [0,255] and mask is binary [0, 1]
        :return: the action parameters to take with the shape [batch_size, action_dim]
        """
        x = x.float()
        image = x[:, 0:1, :, :] / 255.0  # Normalize the image to [0, 1]
        mask = x[:, 1:2, :, :]
        image = self.bn_image(image)
        mask = self.bn_mask(mask)

        x = torch.cat([image, mask], dim=1)

        x = self.feature_extractor(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x)).view(-1, *self.env.action_space.shape)
        output = (x.view(-1, np.prod(self.env.action_space.shape)) * self.action_scale) + self.action_bias
        return output
