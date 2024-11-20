import torch
import torch.nn as nn

class LAuReL_PABlock(nn.Module):
    def __init__(self, in_channels, out_channels, rank=16, num_layers=5):
        """
        LAuReL-PA Block with contributions from previous activations.
        
        Args:
        - in_channels (int): Number of input channels.
        - out_channels (int): Number of output channels.
        - rank (int): Rank 'r' for the low-rank factorization (r << in_channels).
        - num_layers (int): Maximum number of previous layers' activations to consider.
        """
        super(LAuReL_PABlock, self).__init__()
        
        # Transformation path f(x)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Learnable low-rank matrices A and B for h(x)
        self.A = nn.Parameter(torch.randn(out_channels, rank))  # A: D x r
        self.B = nn.Parameter(torch.randn(rank, out_channels))  # B: r x D

        # Learnable weights for previous activations (γ)
        self.gammas = nn.Parameter(torch.ones(num_layers))  # γ_j for j = 0 to num_layers-1

        # Buffer for previous activations
        self.register_buffer('num_stored', torch.tensor(0))
        self.max_layers = num_layers
        self.previous_activations = []

    def forward(self, x):
        # Apply transformation path f(x)
        f_x = self.conv(x)
        f_x = self.bn(f_x)
        f_x = self.relu(f_x)
        
        # Collect contributions from previous activations
        contribution = 0
        for j, prev_x in enumerate(self.previous_activations):
            # Compute low-rank transformation ABx_j
            batch_size, channels, height, width = prev_x.size()
            prev_x_flat = prev_x.view(batch_size, channels, -1)  # Flatten spatial dimensions
            AB_prev_x = torch.matmul(self.A, torch.matmul(self.B, prev_x_flat))
            AB_prev_x = AB_prev_x.view_as(prev_x)  # Reshape back
            
            # Weighted contribution γ_j * ABx_j
            contribution += self.gammas[j] * AB_prev_x
        
        # Store current activation (detached from computation graph)
        if len(self.previous_activations) >= self.max_layers:
            self.previous_activations.pop(0)
        self.previous_activations.append(x.detach().clone())

        # Output: f(x) + Σ γ_j * ABx_j + x
        out = f_x + contribution + x
        return out
    def reset_activations(self):
        """Reset the stored activations"""
        self.previous_activations = []


# Example usage
if __name__ == "__main__":
    # Input tensor (e.g., batch of 8 RGB images, 32x32 size)
    x = torch.randn(8, 32, 32, 32)  # 32 input channels
    
    # Instantiate LAuReL-PA block
    laurel_pa_block = LAuReL_PABlock(in_channels=32, out_channels=32, rank=16, num_layers=3)
    
    # Forward pass
    for _ in range(5):  # Simulate a sequence of layers with previous activations
        output = laurel_pa_block(x)
        print("Output shape:", output.shape)

