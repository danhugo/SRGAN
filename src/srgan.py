import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            use_bn: bool,
            use_act: bool,
            discriminator: bool,
            **kwargs,
            ) -> None:
        super(ConvBlock, self).__init__()
        self.use_act = use_act
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs, bias= not use_bn)
        self.bn_or_identity = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.activation = (
            nn.LeakyReLU(0.2, inplace=True)
            if discriminator 
            else nn.PReLU()
        ) if self.use_act else None
        
    def forward(self, x):
        return self.activation(self.bn_or_identity(self.conv(x))) if self.use_act else self.bn_or_identity(self.conv(x))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super(ResidualBlock, self).__init__()
        self.convblock1 = ConvBlock(in_channels, in_channels, use_bn=True, use_act=True, discriminator=False, kernel_size=3 , stride=1, padding=1)
        self.convblock2 = ConvBlock(in_channels, in_channels, use_bn=True, use_act=False, discriminator=False, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return x + self.convblock2(self.convblock1(x))

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * 4, 3, 1, 1)
        self.bn = nn.BatchNorm2d(in_channels * 4)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()
    
    def forward(self, x):
        return self.prelu(self.pixel_shuffle(self.conv(x)))

class Generator(nn.Module):
    def __init__(
            self, 
            nf: int = 64,
            num_res_blocks: int = 16) -> None:
        """SRGAN Generator.
        - nf (int): number of filter in conv layer of each residual block (default 64).
        - num_res_blocks (int): number of residual blocks (default 5).

        Return super resolution image with x 4 scale.
        """
        super(Generator, self).__init__()
        self.intial = ConvBlock(3, nf, use_bn=False, use_act=True, discriminator=False, kernel_size=9, stride=1, padding=4)
        self.res_blocks = nn.Sequential(*[ResidualBlock(nf) for _ in range(num_res_blocks)])
        self.convblock = ConvBlock(nf, nf, use_bn=True, use_act=False, discriminator=False, kernel_size=3, stride=1, padding=1)
        # an skip connection here

        # upsample x 4
        self.upsample1 = UpsampleBlock(nf)
        self.upsample2 = UpsampleBlock(nf)
        
        self.final = nn.Conv2d(nf, 3, 9, 1, 4)
    
    def forward(self, x):
        initial = self.intial(x)
        out = self.convblock(self.res_blocks(initial))
        out = initial + out
        out = self.upsample1(out)
        out = self.upsample2(out)
        out = self.final(out)
        return out

class Discriminator(nn.Module):
    def __init__(
            self, 
            nf: int = 64) -> None:
        """SRGAN Discriminator
        - nf (int): number of filter of the first conv layer (default 64)

        Return (Tensor.float): posibility whether gen image is real in range [0 - 1].
        """
        super(Discriminator, self).__init__()

        self.convblock1 =ConvBlock(3, nf, use_bn=False, use_act=True, discriminator=True, kernel_size=3, stride=1, padding=1) # in_channels = img_channels 96
        self.convblock2 = ConvBlock(nf, nf, use_bn=True, use_act=True, discriminator=True, kernel_size=3, stride=2, padding=1)

        self.convblock3 = ConvBlock(nf, nf * 2, use_bn=True, use_act=True, discriminator=True, kernel_size=3, stride=1, padding=1)
        self.convblock4 = ConvBlock(nf * 2, nf * 2, use_bn=True, use_act=True, discriminator=True, kernel_size=3, stride=2, padding=1)
        self.convblock5 = ConvBlock(nf * 2, nf * 4, use_bn=True, use_act=True, discriminator=True, kernel_size=3, stride=1, padding=1)
        self.convblock6 = ConvBlock(nf * 4, nf * 4, use_bn=True, use_act=True, discriminator=True, kernel_size=3, stride=2, padding=1)
        self.convblock7 = ConvBlock(nf * 4, nf * 8, use_bn=True, use_act=True, discriminator=True, kernel_size=3, stride=1, padding=1)
        self.convblock8 = ConvBlock(nf * 8, nf * 8, use_bn=True, use_act=True, discriminator=True, kernel_size=3, stride=2, padding=1)
        
        # self.adaavgpool = nn.AdaptiveAvgPool2d((6, 6)) # ensure that every img size would end up with size 6 x 6 not only with size 96 x 96 as paper
        self.dens1 = nn.Linear(nf * 8 * 6 * 6, 1024)
        self.leaky_relu_dens1 = nn.LeakyReLU(0.2, inplace=True)
        self.dens2 = nn.Linear(1024, 1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.convblock1(x)
        out = self.convblock2(out)
        out = self.convblock3(out)
        out = self.convblock4(out)
        out = self.convblock5(out)
        out = self.convblock6(out)
        out = self.convblock7(out)
        out = self.convblock8(out)
        # out = self.adaavgpool(out)
        out = self.dens1(out.view(out.size(0),-1))
        out = self.leaky_relu_dens1(out)
        out = self.dens2(out)
        return out
