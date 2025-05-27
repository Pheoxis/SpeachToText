import torch.nn as nn
import torch.nn.functional as F

class ResidualDownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size=4):
        super().__init__()
        
        # First convolution with downsampling
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding="same"
        )
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Second convolution (maintains dimensions)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        
        # ReLU activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # Main path through the block
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output) + x  # Residual connection
        output = self.conv2(output)
        return output


class DownsamplingNetwork(nn.Module):
    def __init__(
        self,
        embedding_dim=128,
        hidden_dim=64,
        in_channels=1,
        initial_mean_pooling_kernel_size=2,
        strides=[6, 6, 8, 4, 2],
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Initial mean pooling layer
        self.mean_pooling = nn.MaxPool1d(kernel_size=initial_mean_pooling_kernel_size)
        
        # Create residual downsampling blocks
        for i in range(len(strides)):
            self.layers.append(
                ResidualDownSampleBlock(
                    hidden_dim if i > 0 else in_channels,
                    hidden_dim,
                    strides[i],
                    kernel_size=8,
                )
            )
        
        # Final convolution to embedding dimension
        self.final_conv = nn.Conv1d(
            hidden_dim, embedding_dim, kernel_size=4, padding="same"
        )

    def forward(self, x):
        # Apply initial pooling
        x = self.mean_pooling(x)
        
        # Pass through residual blocks
        for layer in self.layers:
            x = layer(x)
        
        # Final convolution
        x = self.final_conv(x)
        x = x.transpose(1, 2)  # Change shape to [batch, length, embedding_dim]
        return x

