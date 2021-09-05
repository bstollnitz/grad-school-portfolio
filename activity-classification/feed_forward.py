import math

import torch.nn as nn

class FeedForward(nn.Module):

    @property
    def loss_function(self):
        return self._loss_function

    def __init__(self, input_size, hidden_size, output_size):
        super(FeedForward, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        self._loss_function = nn.NLLLoss()

        # Initialize the weights and biases.
        for layer in [self.fc1, self.fc2]:
            weight = layer.weight
            stdv = 1.0 / math.sqrt(weight.size(1))
            nn.init.uniform_(weight, -stdv, stdv)
            if layer.bias is not None:
                nn.init.uniform_(layer.bias, -stdv, stdv)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return nn.functional.log_softmax(x, dim=1)
