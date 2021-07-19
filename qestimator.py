import torch
import torch.nn as nn
import torch.nn.functional as F


class AlexNetQEstimator(nn.Module):

    def __init__(self) -> None:
        super(AlexNetQEstimator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=(3, 4), stride=1, padding=(1, 0), padding_mode='circular'),
            # (7, 11) -> (7, 8)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=0),  # (7, 8) -> (5, 6)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),  # (5, 6) -> (3, 4)
            nn.ReLU(inplace=True),
            # nn.Conv2d(128, 64, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(128 * 3 * 4, 1024),
            nn.ReLU(inplace=True),
            # nn.Dropout(),
            # nn.Linear(512, 512),
            # nn.ReLU(inplace=True),
            nn.Linear(1024, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def __str__(self):
        return "{0}".format(AlexNetQEstimator.__name__)


class OneLayerNetQEstimator(nn.Module):

    def __init__(self) -> None:
        super(OneLayerNetQEstimator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, stride=1, padding=1),  # (7, 11) -> (7, 11)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 4), stride=(3, 4), padding=(1, 1), padding_mode='circular'),
            # (7, 11) -> (3, 3)
            nn.ReLU(inplace=True),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(64 * 3 * 3, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def __str__(self):
        return "{0}".format(OneLayerNetQEstimator.__name__)


class TwoLayerNetQEstimator(nn.Module):

    def __init__(self) -> None:
        super(TwoLayerNetQEstimator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1),  # (7, 11) -> (7, 11)
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 32, kernel_size=3, stride=1, padding=1),  # (7, 11) -> (7, 11)
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=(3, 4), stride=(3, 4), padding=(1, 1), padding_mode='circular'),
            # (7, 11) -> (3, 3)
            nn.ReLU(inplace=True),
            # nn.ReLU(inplace=True),
            # nn.Conv2d(64, 64, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(64 * 3 * 3, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def __str__(self):
        return "{0}".format(OneLayerNetQEstimator.__name__)


class GooseNet4(nn.Module):
    def __init__(self, n_channels=14):
        self.n_channels = n_channels
        super(GooseNet4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels,
                               kernel_size=3, padding=(1, 1), padding_mode='circular')
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=2 * n_channels,
                               kernel_size=3, padding=(1, 1), padding_mode='circular')
        self.conv3 = nn.Conv2d(in_channels=2 * n_channels, out_channels=4 * n_channels,
                               kernel_size=3, padding=(1, 1), padding_mode='circular')
        self.final_conv = nn.Conv2d(in_channels=4 * n_channels, out_channels=1,
                                    kernel_size=5, padding=0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        north = x[:, :, 0:5, 3:8]
        east = torch.rot90(x[:, :, 1:6, 4:9], 1, [2, 3])
        south = torch.rot90(x[:, :, 2:7, 3:8], 2, [2, 3])
        west = torch.rot90(x[:, :, 1:6, 2:7], 3, [2, 3])

        # т.к. выходной вектор сети имеет форму [batch_size,1,1,4] - число батчей, число каналов, высота, ширина
        # (последний слой - это сверточная сеть, тут не очень очевидно с выходом, может добавить линейный слой?)
        # убираем лишние оси
        north = self.final_conv(north).squeeze(1).squeeze(2)
        east = self.final_conv(east).squeeze(1).squeeze(2)
        south = self.final_conv(south).squeeze(1).squeeze(2)
        west = self.final_conv(west).squeeze(1).squeeze(2)
        # Склеиваем все результаты в один батч
        return torch.cat([north, east, south, west], 1)

    def __str__(self):
        return "YarEstimator4Layer_{0}ch".format(self.n_channels)


class GooseNetResidual4(nn.Module):
    def __init__(self, n_channels=14):
        self.n_channels = n_channels
        super(GooseNetResidual4, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=n_channels, out_channels=n_channels,
                               kernel_size=3, padding=(1, 1), padding_mode='circular')
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=2 * n_channels,
                               kernel_size=3, padding=(1, 1), padding_mode='circular')
        self.conv3 = nn.Conv2d(in_channels=2 * n_channels, out_channels=4 * n_channels,
                               kernel_size=3, padding=(1, 1), padding_mode='circular')
        self.final_conv = nn.Conv2d(in_channels=4 * n_channels, out_channels=1,
                                    kernel_size=5, padding=0)
        self.proj = nn.Conv2d(in_channels=self.n_channels, out_channels=4 * self.n_channels, kernel_size=3,
                              padding=(1, 1), padding_mode='circular')

    def forward(self, x):
        y = x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = self.proj(y) + F.relu(x)
        north = x[:, :, 0:5, 3:8]
        east = torch.rot90(x[:, :, 1:6, 4:9], 1, [2, 3])
        south = torch.rot90(x[:, :, 2:7, 3:8], 2, [2, 3])
        west = torch.rot90(x[:, :, 1:6, 2:7], 3, [2, 3])

        # т.к. выходной вектор сети имеет форму [batch_size,1,1,4] - число батчей, число каналов, высота, ширина
        # (последний слой - это сверточная сеть, тут не очень очевидно с выходом, может добавить линейный слой?)
        # убираем лишние оси
        north = self.final_conv(north).squeeze(1).squeeze(2)
        east = self.final_conv(east).squeeze(1).squeeze(2)
        south = self.final_conv(south).squeeze(1).squeeze(2)
        west = self.final_conv(west).squeeze(1).squeeze(2)
        # Склеиваем все результаты в один батч
        return torch.cat([north, east, south, west], 1)

    def __str__(self):
        return "YarEstimator4Layer_{0}ch".format(self.n_channels)


class GooseNet3(nn.Module):
    def __init__(self, n_channels=14):
        self.n_channels = n_channels
        super(GooseNet3, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=12, out_channels=n_channels,
        #                        kernel_size=3, padding=(1, 1), padding_mode='circular')
        self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=2 * n_channels,
                               kernel_size=3, padding=(1, 1), padding_mode='circular')
        self.conv3 = nn.Conv2d(in_channels=2 * n_channels, out_channels=4 * n_channels,
                               kernel_size=3, padding=(1, 1), padding_mode='circular')
        self.final_conv = nn.Conv2d(in_channels=4 * n_channels, out_channels=1,
                                    kernel_size=5, padding=0)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        north = x[:, :, 0:5, 3:8]
        east = torch.rot90(x[:, :, 1:6, 4:9], 1, [2, 3])
        south = torch.rot90(x[:, :, 2:7, 3:8], 2, [2, 3])
        west = torch.rot90(x[:, :, 1:6, 2:7], 3, [2, 3])

        # т.к. выходной вектор сети имеет форму [batch_size,1,1,4] - число батчей, число каналов, высота, ширина
        # (последний слой - это сверточная сеть, тут не очень очевидно с выходом, может добавить линейный слой?)
        # убираем лишние оси
        north = self.final_conv(north).squeeze(1).squeeze(2)
        east = self.final_conv(east).squeeze(1).squeeze(2)
        south = self.final_conv(south).squeeze(1).squeeze(2)
        west = self.final_conv(west).squeeze(1).squeeze(2)
        # Склеиваем все результаты в один батч
        return torch.cat([north, east, south, west], 1)

    def __str__(self):
        return "YarEstimator3Layer"


class GooseNet2(nn.Module):
    def __init__(self, n_channels=14):
        self.n_channels = n_channels
        super(GooseNet2, self).__init__()
        # self.conv1 = nn.Conv2d(in_channels=12, out_channels=n_channels,
        #                        kernel_size=3, padding=(1, 1), padding_mode='circular')
        # self.conv2 = nn.Conv2d(in_channels=n_channels, out_channels=2 * n_channels,
        #                        kernel_size=3, padding=(1, 1), padding_mode='circular')
        self.conv3 = nn.Conv2d(in_channels=n_channels, out_channels=2 * n_channels,
                               kernel_size=3, padding=(1, 1), padding_mode='circular')
        self.final_conv = nn.Conv2d(in_channels=2 * n_channels, out_channels=1,
                                    kernel_size=5, padding=0)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        north = x[:, :, 0:5, 3:8]
        east = torch.rot90(x[:, :, 1:6, 4:9], 1, [2, 3])
        south = torch.rot90(x[:, :, 2:7, 3:8], 2, [2, 3])
        west = torch.rot90(x[:, :, 1:6, 2:7], 3, [2, 3])

        # т.к. выходной вектор сети имеет форму [batch_size,1,1,4] - число батчей, число каналов, высота, ширина
        # (последний слой - это сверточная сеть, тут не очень очевидно с выходом, может добавить линейный слой?)
        # убираем лишние оси
        north = self.final_conv(north).squeeze(1).squeeze(2)
        east = self.final_conv(east).squeeze(1).squeeze(2)
        south = self.final_conv(south).squeeze(1).squeeze(2)
        west = self.final_conv(west).squeeze(1).squeeze(2)
        # Склеиваем все результаты в один батч
        return torch.cat([north, east, south, west], 1)

    def __str__(self):
        return "YarEstimator2Layer"


class RavenNet(nn.Module):
    def __init__(self):
        super(RavenNet, self).__init__()
        self.start_conv = nn.Conv2d(in_channels=12, out_channels=50,
                                    kernel_size=3)
        self.conv1 = nn.Conv2d(in_channels=50, out_channels=100,
                               kernel_size=3)
        self.final_conv = nn.Conv2d(in_channels=100, out_channels=1,
                                    kernel_size=3)

    def forward(self, x):
        north = torch.roll(x, 1, 2)
        east = torch.roll(x, -1, 3)
        south = torch.roll(x, -1, 2)
        west = torch.roll(x, 1, 3)

        east = torch.rot90(east, 1, [2, 3])
        south = torch.rot90(south, 2, [2, 3])
        west = torch.rot90(west, 3, [2, 3])

        north = F.relu(self.start_conv(north[:, :, :, 2:9]))
        east = F.relu(self.start_conv(east[:, :, 2:9, :]))
        south = F.relu(self.start_conv(south[:, :, :, 2:9]))
        west = F.relu(self.start_conv(west[:, :, 2:9, :]))

        north = F.relu(self.conv1(north))
        east = F.relu(self.conv1(east))
        south = F.relu(self.conv1(south))
        west = F.relu(self.conv1(west))

        north = F.relu(self.final_conv(north)).squeeze(1).squeeze(2)
        east = F.relu(self.final_conv(east)).squeeze(1).squeeze(2)
        south = F.relu(self.final_conv(south)).squeeze(1).squeeze(2)
        west = F.relu(self.final_conv(west)).squeeze(1).squeeze(2)

        return torch.cat([north, east, south, west], 1)


class GooseNetGoogle(nn.Module):
    def __init__(self, n_channels=14):
        self.n_channels = n_channels
        super(GooseNetGoogle, self).__init__()
        self.inception = Inception(14, 16, 24, 32, 4, 8)
        self.final_conv = nn.Conv2d(in_channels=56, out_channels=1,
                                    kernel_size=5, padding=0)

    def forward(self, x):
        x = F.relu(self.inception(x))

        north = x[:, :, 0:5, 3:8]
        east = torch.rot90(x[:, :, 1:6, 4:9], 1, [2, 3])
        south = torch.rot90(x[:, :, 2:7, 3:8], 2, [2, 3])
        west = torch.rot90(x[:, :, 1:6, 2:7], 3, [2, 3])

        north = self.final_conv(north).squeeze(1).squeeze(2)
        east = self.final_conv(east).squeeze(1).squeeze(2)
        south = self.final_conv(south).squeeze(1).squeeze(2)
        west = self.final_conv(west).squeeze(1).squeeze(2)
        # Склеиваем все результаты в один батч
        return torch.cat([north, east, south, west], 1)

    def __str__(self):
        return "GooseNetGoogle"


# google net
class Inception(nn.Module):
    __constants__ = ['branch2', 'branch3']

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, conv_block=None):
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)

        outputs = [branch1, branch2, branch3]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding_mode='circular', bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
