import torch
import torch.nn as nn
import torch.autograd as autograd

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        return x

class CNNModelBasic(nn.Module):
    """
    CNN basic without dropout
    """
    def __init__(self, num_classes,cfg=[64,64,'A',128,128,'A',256,256,'A',512,512,'A']):
        super(CNNModelBasic,self).__init__()

        # self.conv = nn.Sequential(
        #     ConvBlock(in_channels=3, out_channels=64),
        #     ConvBlock(in_channels=64,out_channels=64),
        #     nn.AvgPool2d(2),
        #     ConvBlock(in_channels=64, out_channels=128),
        #     ConvBlock(in_channels=128, out_channels=128),
        #     nn.AvgPool2d(2),
        #     ConvBlock(in_channels=128, out_channels=256),
        #     ConvBlock(in_channels=256, out_channels=256),
        #     nn.AvgPool2d(2),
        #     ConvBlock(in_channels=256, out_channels=512),
        #     ConvBlock(in_channels=512, out_channels=512),
        #     nn.AvgPool2d(2)
        # )

        self.conv = self.make_layers(cfg)


        self.fc = nn.Sequential(

            nn.Linear(cfg[-2], 128),
            nn.PReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, num_classes),
        )


    def make_layers(self,cfg):
        in_channels = 3
        layers = []
        for v in cfg:
            if v == 'A':
                layers += [nn.AvgPool2d(2)]
            else:
                layers += [ConvBlock(in_channels=in_channels,out_channels=v)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        # print (x.shape) # [x,512,8,8]
        x = torch.mean(x, dim=3)
        x, _ = torch.max(x, dim=2)
        x = self.fc(x)
        return x
