import torch
import torch.nn as nn
import torch.nn.functional as F

class LAuReL_LRBlock(nn.Module):
    def __init__(self, in_channels, out_channels, rank=16):
        """
        LAuReL-LR Block with low-rank learnable transformation.
        
        Args:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - rank (int): Rank 'r' for the low-rank factorization (r << in_channels).
        """
        super(LAuReL_LRBlock, self).__init__()
        
        # Transformation path f(x)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Learnable low-rank matrices A and B
        self.A = nn.Parameter(torch.randn(out_channels, rank))  # A: D x r
        self.B = nn.Parameter(torch.randn(rank, out_channels))  # B: r x D

    def forward(self, x):
        # Transformation path f(x)
        f_x = self.conv(x)
        f_x = self.bn(f_x)
        f_x = self.relu(f_x)
        
        # Low-rank residual connection ABx + x
        batch_size, channels, height, width = x.size()
        x_flat = x.view(batch_size, channels, -1)  # Flatten spatial dimensions
        
        # Compute ABx
        ABx = torch.matmul(self.A, torch.matmul(self.B, x_flat))  # Shape: (batch_size, D, HW)
        ABx = ABx.view_as(x)  # Reshape back to (batch_size, channels, height, width)
        
        # Output: f(x) + ABx + x
        out = f_x + ABx + x
        return out


# Example usage
if __name__ == "__main__":
    # Input tensor (e.g., batch of 8 RGB images, 32x32 size)
    x = torch.randn(8, 32, 32, 32)  # 32 input channels
    
    # Instantiate LAuReL-LR block
    laurel_lr_block = LAuReL_LRBlock(in_channels=32, out_channels=32, rank=16)
    
    # Forward pass
    output = laurel_lr_block(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
