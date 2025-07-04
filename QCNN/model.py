import torch.nn as nn
import torch
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(3, 6, 5),
                                 nn.MaxPool2d(2, 2),
                                 nn.Conv2d(6, 16, 5),
                                 nn.MaxPool2d(2, 2),
                                 nn.Flatten(),
                                 nn.Linear(16 * 5 * 5, 120),
                                 nn.ReLU(),
                                 nn.Linear(120, 84),
                                 nn.ReLU(),
                                 nn.Linear(84, 10)
                                 )

    def forward(self, x):
        return self.net(x)

