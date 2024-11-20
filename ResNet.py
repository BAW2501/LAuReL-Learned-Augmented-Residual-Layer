from torch import nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        Standard ResNet block for comparison with LAuReL blocks.
        
        Args:
        - in_channels (int): Number of input channels
        - out_channels (int): Number of output channels
        """
        super(ResBlock, self).__init__()
        
        # Main transformation path
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Store input for residual connection
        identity = x
        
        # Apply transformation path f(x)
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        
        # Add residual connection
        out = out + identity
        
        return out