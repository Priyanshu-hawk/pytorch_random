import torch
from torch import nn

class ResidualBlockResNet50Above(nn.Module):
    def __init__(self, in_chan, interm_chan, identiti_downsample=None, stride = 1) -> None:
        super().__init__()
        self.expansion = 4
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_chan, interm_chan, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(interm_chan),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(interm_chan, interm_chan, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(interm_chan),
            nn.ReLU(),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(interm_chan, interm_chan*self.expansion, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(interm_chan*self.expansion)
        )
        self.identiti_downsample = identiti_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        identiti = x.clone()

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if self.identiti_downsample is not None:
            identiti = self.identiti_downsample(identiti)
        
        x += identiti
        x = self.relu(x)
        return x

class ResNet50Above(nn.Module):
    def __init__(self, Residual_block , layers, num_classes) -> None:
        """
        Residual block
        layer list
        out classes
        """
        super().__init__()
        self.in_channels = 64
        self.base_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7,stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self._make_layer(Residual_block, layers[0], inter_chan=64, stride=1) 
        self.layer2 = self._make_layer(Residual_block, layers[1], inter_chan=128, stride=2) 
        self.layer3 = self._make_layer(Residual_block, layers[2], inter_chan=256, stride=2) 
        self.layer4 = self._make_layer(Residual_block, layers[3], inter_chan=512, stride=2) 

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.base_layer(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


    def _make_layer(self, block, no_res_block, inter_chan, stride):
        indentit_downsample = None
        layers = []
        if stride != 1 or self.in_channels != inter_chan*4:
            indentit_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    inter_chan * 4,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(inter_chan * 4),
            )
        
        layers.append(block(self.in_channels, inter_chan, indentit_downsample, stride))

        self.in_channels = inter_chan*4

        for i in range(no_res_block-1):
            layers.append(block(self.in_channels, inter_chan))

        return nn.Sequential(*layers)
