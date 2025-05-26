import torch.nn as nn
import torch.nn.functional as F

class ResidualDownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size=4):
        super().__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding="same"
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size//2  # Use explicit padding instead of "same"
        )
        
        # Projection layer for residual connection when dimensions don't match
        self.residual_projection = None
        if in_channels != out_channels or stride != 1:
            self.residual_projection = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, stride=stride, padding=0
            )
        
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        
        # First convolution
        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output)
        
        # Second convolution with stride
        output = self.conv2(output)
        
        # Handle residual connection
        if self.residual_projection is not None:
            residual = self.residual_projection(residual)
        
        # Ensure dimensions match before adding
        if output.shape != residual.shape:
            # Adjust residual to match output dimensions
            min_length = min(output.shape[2], residual.shape[2])
            output = output[:, :, :min_length]
            residual = residual[:, :, :min_length]
        
        # Add residual connection
        output = output + residual
        output = self.relu(output)
        
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
        
        # Use AvgPool1d instead of MaxPool1d for better gradient flow
        self.initial_pooling = nn.AvgPool1d(kernel_size=initial_mean_pooling_kernel_size)
        
        for i in range(len(strides)):
            self.layers.append(
                ResidualDownSampleBlock(
                    hidden_dim if i > 0 else in_channels,
                    hidden_dim,
                    strides[i],
                    kernel_size=8,
                )
            )
        
        self.final_conv = nn.Conv1d(
            hidden_dim, embedding_dim, kernel_size=4, padding="same"
        )

    def forward(self, x):
        # Ensure input is 3D: [batch_size, channels, sequence_length]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.initial_pooling(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_conv(x)
        x = x.transpose(1, 2)  # Convert to [batch_size, sequence_length, embedding_dim]
        
        return x
