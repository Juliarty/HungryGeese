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
            nn.Conv2d(12, 32, kernel_size=3, stride=1, padding=1),      # (7, 11) -> (7, 11)
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


class GooseNet(nn.Module):
    def __init__(self):
        n_channels = 12
        super(GooseNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=12, out_channels=n_channels,
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
        return "YarEstimator"


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
