from imports import *

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        return out
# ============================================================================
# RESUNET ARCHITECTURE
# ============================================================================
class ResUNetDeeper(nn.Module):
    """
    SIMPLE FIX: 2 ResBlocks per level instead of 1
    """
    def __init__(self, in_channels=3, num_classes=1, base_filters=64):
        super(ResUNetDeeper, self).__init__()
        
        # Initial convolution
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels, base_filters, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True)
        )
        # Drop out
        self.dropout = nn.Dropout2d(p = 0.2)
        # Encoder - 2 BLOCKS PER LEVEL (this is the key change!)
        self.encoder1_1 = ResidualBlock(base_filters, base_filters)
        self.encoder1_2 = ResidualBlock(base_filters, base_filters)  # NEW
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2_1 = ResidualBlock(base_filters, base_filters * 2)
        self.encoder2_2 = ResidualBlock(base_filters * 2, base_filters * 2)  # NEW
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3_1 = ResidualBlock(base_filters * 2, base_filters * 4)
        self.encoder3_2 = ResidualBlock(base_filters * 4, base_filters * 4)  # NEW
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4_1 = ResidualBlock(base_filters * 4, base_filters * 8)
        self.encoder4_2 = ResidualBlock(base_filters * 8, base_filters * 8)  # NEW
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bridge - 2 BLOCKS
        self.bridge1 = ResidualBlock(base_filters * 8, base_filters * 16)
        self.bridge2 = ResidualBlock(base_filters * 16, base_filters * 16)  # NEW
        
        # Decoder - 2 BLOCKS PER LEVEL
        self.upconv4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.decoder4_1 = ResidualBlock(base_filters * 16, base_filters * 8)
        self.decoder4_2 = ResidualBlock(base_filters * 8, base_filters * 8)  # NEW
        
        self.upconv3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.decoder3_1 = ResidualBlock(base_filters * 8, base_filters * 4)
        self.decoder3_2 = ResidualBlock(base_filters * 4, base_filters * 4)  # NEW
        
        self.upconv2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.decoder2_1 = ResidualBlock(base_filters * 4, base_filters * 2)
        self.decoder2_2 = ResidualBlock(base_filters * 2, base_filters * 2)  # NEW
        
        self.upconv1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.decoder1_1 = ResidualBlock(base_filters * 2, base_filters)
        self.decoder1_2 = ResidualBlock(base_filters, base_filters)  # NEW
        
        # Output
        self.output = nn.Conv2d(base_filters, num_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x = self.input_layer(x)
        
        enc1 = self.encoder1_1(x)
        enc1 = self.encoder1_2(enc1)  # Second block
        x = self.pool1(enc1)
        
        enc2 = self.encoder2_1(x)
        enc2 = self.encoder2_2(enc2)  # Second block
        x = self.pool2(enc2)
        
        enc3 = self.encoder3_1(x)
        enc3 = self.encoder3_2(enc3)  # Second block
        x = self.pool3(enc3)
        
        enc4 = self.encoder4_1(x)
        enc4 = self.encoder4_2(enc4)  # Second block
        x = self.pool4(enc4)
        
        # Bridge
        x = self.bridge1(x)
        x = self.bridge2(x)  # Second block

        x = self.dropout(x)
        # Decoder with skip connections
        x = self.upconv4(x)
        x = torch.cat([x, enc4], dim=1)
        x = self.decoder4_1(x)
        x = self.decoder4_2(x)  # Second block
        
        x = self.upconv3(x)
        x = torch.cat([x, enc3], dim=1)
        x = self.decoder3_1(x)
        x = self.decoder3_2(x)  # Second block
        
        x = self.upconv2(x)
        x = torch.cat([x, enc2], dim=1)
        x = self.decoder2_1(x)
        x = self.decoder2_2(x)  # Second block
        
        x = self.upconv1(x)
        x = torch.cat([x, enc1], dim=1)
        x = self.decoder1_1(x)
        x = self.decoder1_2(x)  # Second block
            
        x = self.output(x)
        return x