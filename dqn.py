import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, c, h, w, kernel_size, stride, padding, dilation, outputs, device):
        super(DQN, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(c, 24,
                               kernel_size=kernel_size,
                               stride=stride,
                               dilation=dilation,
                               padding=padding,
                               padding_mode='zeros')

        self.bn1 = nn.BatchNorm2d(24)
        self.conv2 = nn.Conv2d(24, 48,
                               kernel_size=kernel_size,
                               stride=stride,
                               dilation=dilation,
                               padding=padding,
                               padding_mode='zeros')
        self.bn2 = nn.BatchNorm2d(48)
        self.conv3 = nn.Conv2d(48, 48,
                               kernel_size=kernel_size,
                               stride=stride,
                               dilation=dilation,
                               padding=padding,
                               padding_mode='zeros')

        self.bn3 = nn.BatchNorm2d(48)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size):
            return (size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        linear_input_size = convw * convh * 48
        self.head = nn.Linear(linear_input_size, outputs)
    # Called with either one element to determine next action, or a batch

    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))
