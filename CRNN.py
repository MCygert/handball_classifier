import torch
from torch import nn




classes = ('pass', 'shot', 'save')


class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        self.conv_layer_1 = self._create_convolution_layer(15, 32)
        self.conv_layer_2 = self._create_convolution_layer(32, 64)
        self.fc1 = nn.Linear(262144, 1)
        self.fc2 = nn.Linear(128, len(classes))
        self.relu = nn.LeakyReLU()
        self.batch = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(p=0.15)

    def _create_convolution_layer(self, c_in, c_out):
        return nn.Sequential(
            nn.Conv3d(c_in, c_out, kernel_size=(4, 4, 4), padding=2),
            nn.LeakyReLU(),
            nn.MaxPool3d((2, 2, 2)))

    def forward(self, x):
        out = self.conv_layer_1(x)
        out = self.conv_layer_2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.batch(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out
