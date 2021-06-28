import torch.nn as nn
import torch.nn.functional as F


class QEstimator(nn.Module):

    def __init__(self, c, h, w, outputs, device):
        super(QEstimator, self).__init__()

        self.device = device
        self.kernel_sizes = [(2, 2), (3, 3), (2, 2)]
        self.channels = [64, 32, 16]
        strides = [(1, 1), (1, 1), (1, 1)]
        dilations = [(1, 1), (1, 1), (1, 1)]
        paddings = [(1, 1), (1, 1), (1, 1)]
        padding_mode = 'circular'

        self.conv1 = nn.Conv2d(c, self.channels[0],
                               kernel_size=self.kernel_sizes[0],
                               stride=strides[0],
                               dilation=dilations[0],
                               padding=paddings[0],
                               padding_mode=padding_mode)

        self.bn1 = nn.BatchNorm2d(self.channels[0])
        self.conv2 = nn.Conv2d(self.channels[0], self.channels[1],
                               kernel_size=self.kernel_sizes[1],
                               stride=strides[1],
                               dilation=dilations[1],
                               padding=paddings[1],
                               padding_mode=padding_mode)
        self.bn2 = nn.BatchNorm2d(self.channels[1])
        self.conv3 = nn.Conv2d(self.channels[1], self.channels[2],
                               kernel_size=self.kernel_sizes[2],
                               stride=strides[1],
                               dilation=dilations[1],
                               padding=paddings[1],
                               padding_mode=padding_mode)

        self.bn3 = nn.BatchNorm2d(self.channels[2])

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, padding, dilation, kernel_size, stride):
            return (size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

        conv_h = h
        conv_w = w
        for i in range(3):
            conv_h = conv2d_size_out(conv_h, paddings[i][0], dilations[i][0], self.kernel_sizes[i][0], strides[i][0])
            conv_w = conv2d_size_out(conv_w, paddings[i][1], dilations[i][1], self.kernel_sizes[i][1], strides[i][1])

        linear_input_size = conv_w * conv_h * self.channels[2]
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch

    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

    def __str__(self):
        kernel_str = "ks{0}{1}_{2}{3}_{4}{5}".format(self.kernel_sizes[0][0],
                                                     self.kernel_sizes[0][1],
                                                     self.kernel_sizes[1][0],
                                                     self.kernel_sizes[1][1],
                                                     self.kernel_sizes[2][0],
                                                     self.kernel_sizes[2][1])
        layer_str = "ch{0}_{1}_{2}".format(self.channels[0],
                                           self.channels[1],
                                           self.channels[2])

        return "{0}_{1}_{2}".format(QEstimator.__name__, kernel_str, layer_str)


class QEstimatorFactory:
    def __init__(self, estimator_class, channel_in, height, width, device):
        self.abstract_network_class = estimator_class
        self.channels_in = channel_in
        self.H = height
        self.W = width
        self.outputs = 4
        self.device = device

    def create(self):
        return self.abstract_network_class(c=self.channels_in,
                                           h=self.H,
                                           w=self.W,
                                           outputs=self.outputs,
                                           device=self.device).to(self.device)