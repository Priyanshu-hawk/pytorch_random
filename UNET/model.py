# import torch
# from torch import nn
# from torchinfo import summary

# device = "cuda" if torch.cuda.is_available() else "cpu"
# class UNetCore(nn.Module):
#     def __init__(self, input_dim):
#         super().__init__()
#         self.input_dim = input_dim
        
#     def block(self, x, filter):
#         conv_d1 = nn.Sequential(
#             nn.Conv2d(x.shape[1], filter, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(filter),
#             nn.ReLU(),

#             nn.Conv2d(filter, filter, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(filter),
#             nn.ReLU()
#         )
#         return conv_d1(x)
    
#     def encoder(self, x, filter):
#         x = self.block(x, filter)
#         s = nn.MaxPool2d(kernel_size=2)(x)
#         return x, s

#     def decode(self, unpool_feat, x , filter):
#         s = nn.ConvTranspose2d(x.shape[1], filter, kernel_size=2, stride=2, padding=0)(x)
#         s = torch.concatenate([unpool_feat, s], dim=1)
#         x = self.block(s, filter)
#         return x
    
#     def out_layer(self, x, filter):
#         x = nn.Conv2d(x.shape[1], filter, kernel_size=1, stride=1, padding=0)(x)
#         return x

#     def forward(self, x):
#         c1,ds1 = self.encoder(x, 64)
#         c2,ds2 = self.encoder(ds1, 128)
#         c3,ds3 = self.encoder(ds2, 256)
#         c4,ds4 = self.encoder(ds3, 512)
        
#         c5 = self.block(ds4, 1024)

#         print(c1.shape, ds1.shape)
#         print(c2.shape, ds2.shape)
#         print(c3.shape, ds3.shape)
#         print(c4.shape, ds4.shape)
#         print(c5.shape)
#         print("---")

#         x = self.decode(c4, c5, 512)
#         print(x.shape)
#         x = self.decode(c3, c4, 256)
#         print(x.shape)
#         x = self.decode(c2, c3, 128)
#         print(x.shape)
#         x = self.decode(c1, c2, 64)
#         print(x.shape)
#         x = self.out_layer(x, self.input_dim)
#         return x


# if __name__ == "__main__":
#     model = UNetCore(3)
#     model = model.to(device)
#     summary(model, input_size=(32,3,256,256))
#     # img = torch.randn(32,3,256,256)
#     # print(img.shape)
#     # print(model(img).shape)

import torch
from torch import nn
from torchinfo import summary

device = "cuda" if torch.cuda.is_available() else "cpu"

class UNetCore(nn.Module):
    def __init__(self, input_dim, n_class=1):
        super().__init__()
        self.input_dim = input_dim
        self.n_class = n_class
        
        self.encoder1 = self.conv_block(3, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)

        self.decoder4 = self.conv_block(1024, 512)
        self.decoder3 = self.conv_block(512, 256)
        self.decoder2 = self.conv_block(256, 128)
        self.decoder1 = self.conv_block(128, 64)

        self.upsample4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upsample3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upsample1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.final_conv = nn.Conv2d(64, self.n_class, kernel_size=1)

        self.to(device)

    def conv_block(self, in_channels, out_channels):
        """Creates a Convolutional Block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = x.to(device)  # Ensure input is on the correct device

        # Encoder
        c1 = self.encoder1(x)
        ds1 = self.maxpool(c1)

        c2 = self.encoder2(ds1)
        ds2 = self.maxpool(c2)

        c3 = self.encoder3(ds2)
        ds3 =self.maxpool(c3)

        c4 = self.encoder4(ds3)
        ds4 = self.maxpool(c4)

        # Bottleneck
        c5 = self.bottleneck(ds4)

        # Decoder
        x = self.upsample4(c5)
        x = torch.cat([c4, x], dim=1)
        x = self.decoder4(x)

        x = self.upsample3(x)
        x = torch.cat([c3, x], dim=1)
        x = self.decoder3(x)

        x = self.upsample2(x)
        x = torch.cat([c2, x], dim=1)
        x = self.decoder2(x)

        x = self.upsample1(x)
        x = torch.cat([c1, x], dim=1)
        x = self.decoder1(x)

        x = self.final_conv(x)
        return x


if __name__ == "__main__":
    model = UNetCore(3, 10)
    summary(model, input_size=(32, 3, 256, 256))
