import torch.nn as nn
from torch.nn.utils import weight_norm
import torch


#################  1-D TCN Model by Mustafa et al. ###################################################
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, self.chomp_size:].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, bias=True))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, bias=True))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

#         self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
#                                  self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1, bias=True) if n_inputs != n_outputs else None
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


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=int((kernel_size - 1)/2 * dilation_size), dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

 
class MustafaNet(nn.Module):
    def __init__(self):
        super(MustafaNet, self).__init__()
        self.tcn_local = TemporalConvNet(num_inputs=1, num_channels=[3, 6, 6, 6, 6, 6, 5], kernel_size=9, dropout=0.2)
        self.regression = nn.Conv1d(in_channels=5, out_channels=1, kernel_size=1)

    def forward(self, input):
        #out = self.tcn_local(input[:,:,:,3])  # only uncomment when you are using 2-D dataloaders
        out = self.tcn_local(input)
        out = self.regression(out)
        return out        
    



#################  1-D Physics-based Model by Vishal Das et al. ###################################################
class VishalNet(nn.Module):
    def __init__(self):
        super(VishalNet, self).__init__()
        self.cnn1 = nn.Conv1d(1, 60, 81, 1, 40)
        self.cnn2 = nn.Conv1d(60, 1, 301, 1, 150)
    
    def forward(self, input):
        out1 = nn.functional.relu(self.cnn1(input))
        out2 = self.cnn2(out1)
        return out2    
    
    
    

#################  1-D Physics-based Model by Motaz et al. ###################################################
class inverse_model(nn.Module):
    def __init__(self,resolution_ratio=1,nonlinearity="tanh"):
        super(inverse_model, self).__init__()
        self.resolution_ratio = resolution_ratio
        self.activation =  nn.ReLU() if nonlinearity=="relu" else nn.Tanh()

        self.cnn1 = nn.Sequential(nn.Conv1d(in_channels=1,
                                           out_channels=8,
                                           kernel_size=5,
                                           padding=2,
                                           dilation=1),
                                  nn.GroupNorm(num_groups=1,
                                               num_channels=8))

        self.cnn2 = nn.Sequential(nn.Conv1d(in_channels=1,
                                           out_channels=8,
                                           kernel_size=5,
                                           padding=6,
                                           dilation=3),
                                  nn.GroupNorm(num_groups=1,
                                               num_channels=8))

        self.cnn3 = nn.Sequential(nn.Conv1d(in_channels=1,
                                           out_channels=8,
                                           kernel_size=5,
                                           padding=12,
                                           dilation=6),
                                  nn.GroupNorm(num_groups=1,
                                               num_channels=8))

        self.cnn = nn.Sequential(self.activation,
                                 nn.Conv1d(in_channels=24,
                                           out_channels=16,
                                           kernel_size=3,
                                           padding=1),
                                 nn.GroupNorm(num_groups=1,
                                              num_channels=16),
                                 self.activation,

                                 nn.Conv1d(in_channels=16,
                                           out_channels=16,
                                           kernel_size=3,
                                           padding=1),
                                 nn.GroupNorm(num_groups=1,
                                              num_channels=16),
                                 self.activation,

                                 nn.Conv1d(in_channels=16,
                                           out_channels=16,
                                           kernel_size=1),
                                 nn.GroupNorm(num_groups=1,
                                              num_channels=16),
                                 self.activation)

        self.gru = nn.GRU(input_size=1,
                          hidden_size=8,
                          num_layers=3,
                          batch_first=True,
                          bidirectional=True)

        self.up = nn.Sequential(nn.Conv1d(in_channels=16,
                                          out_channels=8,
                                          stride=1,
                                          kernel_size=3,
                                          padding=1),
                                nn.GroupNorm(num_groups=1,
                                             num_channels=8),
                                self.activation,

                                nn.Conv1d(in_channels=8,
                                          out_channels=8,
                                          stride=1,
                                          kernel_size=3,
                                          padding=1),
                                nn.GroupNorm(num_groups=1,
                                             num_channels=8),
                                self.activation)

        self.gru_out = nn.GRU(input_size=8,
                              hidden_size=8,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=True)
        self.out = nn.Linear(in_features=16, out_features=1)


        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        cnn_out1 = self.cnn1(x)
        cnn_out2 = self.cnn2(x)
        cnn_out3 = self.cnn3(x)
        cnn_out = self.cnn(torch.cat((cnn_out1,cnn_out2,cnn_out3),dim=1))

        tmp_x = x.transpose(-1, -2)
        rnn_out, _ = self.gru(tmp_x)
        rnn_out = rnn_out.transpose(-1, -2)

        x = rnn_out + cnn_out
        x = self.up(x)
        tmp_x = x.transpose(-1, -2)
        x, _ = self.gru_out(tmp_x)
        x = self.out(x)
        x = x.transpose(-1,-2)
        return x



class forward_model(nn.Module):
    def __init__(self,resolution_ratio=1,nonlinearity="tanh"):
        super(forward_model, self).__init__()
        self.resolution_ratio = resolution_ratio
        self.activation =  nn.ReLU() if nonlinearity=="relu" else nn.Tanh()
        self.cnn = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=4, kernel_size=9, padding=4),
                                 self.activation,
                                 nn.Conv1d(in_channels=4, out_channels=4,kernel_size=7, padding=3),
                                 self.activation,
                                 nn.Conv1d(in_channels=4, out_channels=1,kernel_size=3, padding=1))


        self.wavelet = nn.Conv1d(in_channels=1,
                             out_channels=1,
                             stride=self.resolution_ratio,
                             kernel_size=51,
                             padding=25)

    def forward(self, x):
        x = self.cnn(x)
        x = self.wavelet(x)
        return x
     

    