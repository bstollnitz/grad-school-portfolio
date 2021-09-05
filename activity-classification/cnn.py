import math

import torch.nn as nn

class CNN(nn.Module):

    @property
    def loss_function(self):
        return self._loss_function

    def __init__(self) -> None:
        super(CNN, self).__init__()

        # Each batch has shape:
        # (batch_size, num_channels, height, width) = (batch_size, 9, 128, 128).

        self.conv1 = nn.Conv2d(
            in_channels = 9, # number of channels
            out_channels = 9, # number of filters
            kernel_size = 3,
            stride = 1,
            padding = 1 # this stride and padding don't change the (height, width) dims
            )
        self.maxpool = nn.MaxPool2d(kernel_size = 2) 
        self.conv2 = nn.Conv2d(
            in_channels = 9, # because we have 9 filters in conv1
            out_channels = 9, # number of filters
            kernel_size = 3,
            stride = 1,
            padding = 1
            )
        self.fc1 = nn.Linear(9*32*32, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 6)

        # Cross entropy loss = softmax + negative log likelihood.
        self._loss_function = nn.CrossEntropyLoss()

        # Initialize the weights and biases.
        for layer in [self.conv1, self.conv2]:
            weight = layer.weight
            nn.init.xavier_uniform_(weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        for layer in [self.fc1, self.fc2, self.fc3]:
            weight = layer.weight
            stdv = 1.0 / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stdv, stdv)
            if layer.bias is not None:
                nn.init.uniform_(layer.bias, -stdv, stdv)

        print(f'  {self}')


    def forward(self, x):
        # Each data point has dims 9 x 128 x 128 = 147,456.

        # Layer 1.
        x = nn.functional.relu(self.conv1(x))
        # Each data point has dims 9 x 128 x 128 = 147,456.
        x = self.maxpool(x)
        # Each data point has dims 9 x 64 x 64 = 36,864.

        # Layer 2.
        x = nn.functional.relu(self.conv2(x))
        # Each data point has dims 9 x 64 x 64 = 36,864.
        x = self.maxpool(x)
        # Each data point has dims 9 x 32 x 32 = 9,216.

        # Reshape data from (9, 32, 32) to (1, 9216).
        x = x.view(-1, 9*32*32)

        # Layer 3.
        x = nn.functional.relu(self.fc1(x))
        # Each data point has dims a 1,000 horizontal vector.

        # Layer 4.
        x = nn.functional.relu(self.fc2(x))
        # Each data point is now a 100 horizontal vector.

        # Layer 5.
        x = self.fc3(x)
        # Each data point is now a 6-long horizontal vector.
        # We don't call softmax because we'll use CrossEntropyLoss, which 
        # calls softmax already.

        return x
