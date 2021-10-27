import torch
from torch import nn

classes = ('pass', 'shot', 'save')


class CRNN(nn.Module):
    def __init__(self):
        super(CRNN, self).__init__()
        self.conv_layer_1 = self._create_convolution_layer(3, 64)
        self.conv_layer_2 = self._create_convolution_layer(64, 128)
        self.fc1 = nn.Linear(6272, 2523)
        self.fc2 = nn.Linear(2523, len(classes))
        self.relu = nn.LeakyReLU()
        self.drop = nn.Dropout(p=0.15)

    def _create_convolution_layer(self, c_in, c_out):
        return nn.Sequential(
            nn.Conv3d(c_in, c_out, kernel_size=4, padding=(3,3,3), stride=(1,2,2)),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool3d((3, 3, 3)))

    def forward(self, x):
        out = self.conv_layer_1(x)
        out = self.conv_layer_2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.drop(out)
        out = self.fc2(out)
        return out
