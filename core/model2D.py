# Define 2-D Model
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp2d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp2d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size, :].contiguous()


class TemporalBlock2d(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock2d, self).__init__()
        self.conv1 = weight_norm(nn.Conv2d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, bias=True))
        self.chomp1 = Chomp2d(padding[0])
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv2d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, bias=True))
        self.chomp2 = Chomp2d(padding[0])
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

#         self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
#                                  self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv2d(n_inputs, n_outputs, (1, 1), bias=True) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet2d(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet2d, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock2d(in_channels, out_channels, kernel_size, stride=1, dilation=(dilation_size, 1),
                                       padding=(int((kernel_size[0]-1)/2 * dilation_size), 1), dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Model2D(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(Model2D, self).__init__()
        self.tcn = TemporalConvNet2d(input_size, num_channels, kernel_size=(kernel_size, 3), dropout=dropout)
        self.groupnorm = nn.GroupNorm(num_channels=num_channels[-1], num_groups=1)
        self.conv = nn.Sequential(nn.Conv2d(in_channels=num_channels[-1],
                                            out_channels=20,
                                            kernel_size=(1, 3),
                                            padding=(0, 1)),
                                  nn.ReLU(),
                                  nn.GroupNorm(num_channels=20, num_groups=1),
                                  nn.Conv2d(in_channels=20,
                                            out_channels=10,
                                            kernel_size=(1, 3),
                                            padding=(0, 1)),
                                  nn.ReLU(),
                                  nn.GroupNorm(num_channels=10, num_groups=1),
                                  nn.Conv2d(in_channels=10,
                                            out_channels=1,
                                            kernel_size=(1, 7),
                                            padding=(0, 0)))

        self.synthesis = nn.Sequential(weight_norm(nn.Conv2d(in_channels=num_channels[-1],
                                                             out_channels=5,
                                                             kernel_size=(5, 3),
                                                             padding=(2, 1))),
                                       nn.ReLU(),
                                       nn.GroupNorm(num_channels=5,
                                                    num_groups=1),
                                       weight_norm(nn.Conv2d(in_channels=5,
                                                             out_channels=10,
                                                             kernel_size=(5, 3),
                                                             padding=(2, 1))),
                                       nn.ReLU(),
                                       nn.GroupNorm(num_channels=10,
                                                    num_groups=1),
                                       weight_norm(nn.Conv2d(in_channels=10,
                                                             out_channels=1,
                                                             kernel_size=(5, 3),
                                                             padding=(2, 1))))

    def forward(self, inputs):
        """Inputs have to have dimension (N, C_in, L_in)"""
        y = self.tcn(inputs)  # input should have dimension (N, C, H, W)
        y = self.groupnorm(y)
        out = self.conv(y).squeeze(1).transpose(1, 2)
        x_hat = self.synthesis(y)
        return out, x_hat


