
import torch
import torch.nn as nn
from torch.nn.functional import softmax

class LAuReLBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        LAuReL Block with learnable weights (α and β) normalized via softmax.
        """
        super(LAuReLBlock, self).__init__()
        # Define the transformation path (f(x))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Learnable parameters for α and β
        self.weights = nn.Parameter(torch.ones(2))  # [α, β], initialized to 1

    def forward(self, x):
        # Apply transformation path f(x)
        f_x = self.conv(x)
        f_x = self.bn(f_x)
        f_x = self.relu(f_x)
        
        # Normalize α and β with softmax
        alpha, beta = softmax(self.weights, dim=0)
        
        # Compute the output as α * f(x) + β * x
        out = alpha * f_x + beta * x
        return out


# Example usage
if __name__ == "__main__":
    # Input tensor (e.g., a batch of 8 RGB images, 32x32 size)
    x = torch.randn(8, 3, 32, 32)
    
    # Instantiate the LAuReL block
    laurel_block = LAuReLBlock(in_channels=3, out_channels=3)
    
    # Forward pass
    output = laurel_block(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
