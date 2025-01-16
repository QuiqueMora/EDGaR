import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

class conv_block(nn.Module):
    def __init__(self, in_channels, feat_dim, n_convs=2) -> None:
        super().__init__()

        self.activation = nn.ReLU()
        for i in range(n_convs):
            self.add_module(f"convolution_{i}",
                            nn.Conv2d(in_channels=in_channels,
                                      out_channels=feat_dim,
                                      kernel_size=3,
                                      padding=1))

    def forward(self, input):

        for conv in self._modules.values():
            x = conv(input)
            x = self.activation(x)
        return x

class encoder_block(nn.Module):
    def __init__(self, in_channels, feat_dim) -> None:
        super().__init__()
        self.conv = conv_block(in_channels=in_channels, feat_dim=feat_dim)
        self.pool = nn.MaxPool2d((2,2))

    def forward(self, input):
        skip = self.conv(input)
        x = self.pool(skip)

        return x, skip

class decoder_block(nn.Module):
    def __init__(self, in_channels, feat_dim) -> None:
        super().__init__()

        self.up_samp = nn.ConvTranspose2d(in_channels=in_channels,
                                          out_channels=feat_dim,
                                          kernel_size=(2,2),
                                          stride=2,
                                          padding=0)
        self.convs = conv_block(in_channels=in_channels,
                                feat_dim=feat_dim)


    def forward(self, input, skip):
        x = self.up_samp(input)
        x = torch.cat((x, skip), dim=1)
        x = self.convs(x)

        return x


class unet(nn.Module):

    def __init__(self):
        super().__init__()

        self.e_1 = encoder_block(4, 64) 
        self.e_2 = encoder_block(64, 128)
        self.e_3 = encoder_block(128, 256)
        self.e_4 = encoder_block(256, 512)
        
        self.conv = conv_block(512, 1024)

        self.d_1 = decoder_block(1024, 512) 
        self.d_2 = decoder_block(512, 256)
        self.d_3 = decoder_block(256, 128)
        self.d_4 = decoder_block(128, 64)
        
        self.last = nn.Conv2d(in_channels=64,
                              out_channels=3,
                              kernel_size=1,
                              padding=1)
        self.activation = nn.ReLU()

    def forward(self, input):
        x, skip_1 = self.e_1(input)
        x, skip_2 = self.e_2(x)
        x, skip_3 = self.e_3(x)
        x, skip_4 = self.e_4(x)

        x = self.conv(x)

        x = self.d_1(x, skip_4)
        x = self.d_2(x, skip_3)
        x = self.d_3(x, skip_2)
        x = self.d_4(x, skip_1)

        x = self.last(x)
        return self.activation(x)

if __name__ == "__main__":
    device = torch.device("cuda")
    # (batch, channels, height, width)

    test = torch.rand((32, 4,256,256),device="cuda")
    model = unet().to(device)
    device=torch.device("cuda")

    model(test)
