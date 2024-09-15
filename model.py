import torch
import torch.nn as nn

# Defining Inception Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else None

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        if self.bn1:
            out = self.bn1(out)
        out = self.relu(out)
        out = out + residual
        out = self.relu(out)
        return out

# Defining Residual-in-Residual Block
class ResidualInResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm=False):
        super(ResidualInResidualBlock, self).__init__()
        self.res1 = ResidualBlock(in_channels, out_channels, batch_norm)
        self.res2 = ResidualBlock(out_channels, out_channels, batch_norm)
        # self.res3 = ResidualBlock(out_channels, out_channels, batch_norm)

    def forward(self, x):
        out = self.res1(x)
        out = self.res2(out)
        # out = self.res3(out)
        return out + x

# Defining the U-Net architecture
class UNetSRx4RiR(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_features=64, dropout_rate=0.0):
        super(UNetSRx4RiR, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.encoder1 = ResidualInResidualBlock(num_features, num_features, batch_norm=True)
        self.downsample1 = self.downsample(num_features, num_features * 2, dropout_rate)

        self.encoder2 = ResidualInResidualBlock(num_features * 2, num_features * 2, batch_norm=True)
        self.downsample2 = self.downsample(num_features * 2, num_features * 4, dropout_rate)

        self.encoder3 = ResidualInResidualBlock(num_features * 4, num_features * 4, batch_norm=True)
        self.downsample3 = self.downsample(num_features * 4, num_features * 8, dropout_rate)

        self.encoder4 = ResidualInResidualBlock(num_features * 8, num_features * 8, batch_norm=True)
        self.downsample4 = self.downsample(num_features * 8, num_features * 16, dropout_rate)

        # Bottleneck
        self.bottleneck = ResidualInResidualBlock(num_features * 16, num_features * 16, batch_norm=True)

        # Decoder
        
        self.upconv4 = nn.ConvTranspose2d(num_features * 16, num_features * 8, kernel_size=2, stride=2)
        self.decoder4 = ResidualInResidualBlock(num_features * 16, num_features * 16, batch_norm=True)
        self.conv3 = nn.Conv2d(num_features * 16, num_features * 8, kernel_size=3, padding=1)
        
        self.upconv3 = nn.ConvTranspose2d(num_features * 8, num_features * 4, kernel_size=2, stride=2)
        self.decoder3 = ResidualInResidualBlock(num_features * 8, num_features * 8, batch_norm=True)
        self.conv4 = nn.Conv2d(num_features * 8, num_features * 4, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(num_features * 4, num_features * 2, kernel_size=2, stride=2)
        self.decoder2 = ResidualInResidualBlock(num_features * 4, num_features * 4, batch_norm=True)
        self.conv5 = nn.Conv2d(num_features * 4, num_features * 2, kernel_size=3, padding=1)

        self.upconv1 = nn.ConvTranspose2d(num_features * 2, num_features, kernel_size=2, stride=2)
        self.decoder1 = ResidualInResidualBlock(num_features * 2, num_features * 2, batch_norm=True)

        # Final output layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(num_features * 2, num_features, kernel_size=3, padding=1),
            nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def downsample(self, in_channels, out_channels, dropout_rate, batch_norm=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=2)]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        # First upscale x4 using bicubic interpolation
        x = torch.nn.functional.interpolate(x, scale_factor=4, mode='bicubic', align_corners=False)

        # Encoder
        enc1 = self.conv1(x) 
        enc1 = self.encoder1(enc1) 

        enc2 = self.downsample1(enc1) 
        enc2 = self.encoder2(enc2)

        enc3 = self.downsample2(enc2)
        enc3 = self.encoder3(enc3) 

        enc4 = self.downsample3(enc3)
        enc4 = self.encoder4(enc4)

        # Bottleneck
        bottleneck = self.downsample4(enc4)
        bottleneck = self.bottleneck(bottleneck)

        # Decoder
        dec4 = self.upconv4(bottleneck) 
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec4 = self.conv3(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec3 = self.conv4(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec2 = self.conv5(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        sr = self.final_conv(dec1)

        return sr