import torch
import torch.nn as nn


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

        self.e_1 = encoder_block(3, 64) 
        self.e_2 = encoder_block(64, 128)
        self.e_3 = encoder_block(128, 256)
        self.e_4 = encoder_block(256, 512)
        
        # 512 + 2 new channels for gaze target
        self.conv = conv_block(514, 1024)

        self.d_1 = decoder_block(1024, 512) 
        self.d_2 = decoder_block(512, 256)
        self.d_3 = decoder_block(256, 128)
        self.d_4 = decoder_block(128, 64)
        
        self.last = nn.Conv2d(in_channels=64,
                              out_channels=4,
                              kernel_size=1,
                              padding=0)
        self.activation = nn.ReLU()
        self.activation_alpha = nn.Sigmoid()

    # @torch.compile
    def forward(self, input_img, target_gaze):
        x, skip_1 = self.e_1(input_img)
        x, skip_2 = self.e_2(x)
        x, skip_3 = self.e_3(x)
        x, skip_4 = self.e_4(x)

        # append gaze to input image as new channels
        target_gaze = target_gaze.view(-1,2,1,1).expand(-1, 2, 14, 14)
        x = torch.cat([x, target_gaze], dim=1)
        x = self.conv(x)

        x = self.d_1(x, skip_4)
        x = self.d_2(x, skip_3)
        x = self.d_3(x, skip_2)
        x = self.d_4(x, skip_1)

        x = self.last(x)
        # this way it can learn what regions stay the same
        # by using the alpha channel to safe parts
        alpha = self.activation_alpha(x[:, 3:4, ...] )
        color = self.activation(x[:, :3, ...])

        return alpha * input_img + (1-alpha) * color

class unet_vanilla(nn.Module):

    def __init__(self):
        super().__init__()

        self.e_1 = encoder_block(5, 64) 
        self.e_2 = encoder_block(64, 128)
        self.e_3 = encoder_block(128, 256)
        self.e_4 = encoder_block(256, 512)
        
        # 512 + 2 new channels for gaze target
        self.conv = conv_block(512, 1024)

        self.d_1 = decoder_block(1024, 512) 
        self.d_2 = decoder_block(512, 256)
        self.d_3 = decoder_block(256, 128)
        self.d_4 = decoder_block(128, 64)
        
        self.last = nn.Conv2d(in_channels=64,
                              out_channels=4,
                              kernel_size=1,
                              padding=0)
        self.activation = nn.ReLU()
        self.activation_alpha = nn.Sigmoid()

    # @torch.compile
    def forward(self, input_img, target_gaze):
        # extract height and width
        # input_img shape is (B, C, H, W)
        h,w = input_img.shape[-2:]
        # expand 2D gaze to size of image
        target_gaze = target_gaze.view(-1, 2,1,1).expand(-1, 2,h,w)
        # append gaze direction as new channel 
        input = torch.cat((input_img, target_gaze), dim=1)

        x, skip_1 = self.e_1(input)
        x, skip_2 = self.e_2(x)
        x, skip_3 = self.e_3(x)
        x, skip_4 = self.e_4(x)

        # append gaze to input image as new channels
        x = self.conv(x)

        x = self.d_1(x, skip_4)
        x = self.d_2(x, skip_3)
        x = self.d_3(x, skip_2)
        x = self.d_4(x, skip_1)

        x = self.last(x)
        # this way it can learn what regions stay the same
        # by using the alpha channel to safe parts
        alpha = self.activation_alpha(x[:, 3:4, ...] )
        color = self.activation(x[:, :3, ...])

        return alpha * input_img + (1-alpha) * color, alpha

if __name__ == "__main__":
    device = torch.device("cuda")
    # (batch, channels, height, width)

    batch = 10
    size= 224
    model = unet_vanilla().to(device)
    device=torch.device("cuda")
    for i in range(300):
        test = torch.rand((batch, 3, size+i, size+i),device="cuda")
        target_gaze = torch.rand((batch,2,),device="cuda")
        try:
            model(test, target_gaze)
            print(size+i, "works")
        except:
            continue
