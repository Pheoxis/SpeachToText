import torch
import torch.nn as nn

from downsampling import DownsamplingNetwork
from rvq import ResidualVectorQuantizer
from self_attention import Transformer


class TranscribeModel(nn.Module):
    def __init__(
        self,
        num_codebooks: int,
        codebook_size: int,
        embedding_dim: int,
        vocab_size: int,
        strides: list[int],
        initial_mean_pooling_kernel_size: int,
        num_transformer_layers: int,
        max_seq_length: int = 2000,
    ):
        super().__init__()
        self.options = {
            "num_codebooks": num_codebooks,
            "codebook_size": codebook_size,
            "embedding_dim": embedding_dim,
            "vocab_size": vocab_size,
            "strides": strides,
            "num_transformer_layers": num_transformer_layers,
            "initial_mean_pooling_kernel_size": initial_mean_pooling_kernel_size,
            "max_seq_length": max_seq_length,
        }
        self.downsampling_network = DownsamplingNetwork(
            embedding_dim=embedding_dim,
            hidden_dim=embedding_dim // 2,
            strides=strides,
            initial_mean_pooling_kernel_size=initial_mean_pooling_kernel_size,
        )
        self.pre_rvq_transformer = Transformer(
        embedding_dim,
        num_layers=num_transformer_layers,
        max_seq_length=max_seq_length,
    )

        self.rvq = ResidualVectorQuantizer(num_codebooks, codebook_size, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor):
        loss = torch.tensor(0.0)
        
        # Remove this line - it's adding an unnecessary dimension
        # x = x.unsqueeze(1)
        
        # Instead, check the input shape and handle it properly
        if x.dim() == 2:  # If input is [batch_size, sequence_length]
            x = x.unsqueeze(1)  # Add channel dimension -> [batch_size, 1, sequence_length]
        elif x.dim() == 3:  # If input is already [batch_size, channels, sequence_length]
            pass  # Keep as is
        else:
            raise ValueError(f"Expected 2D or 3D input tensor, got {x.dim()}D tensor with shape {x.shape}")
        
        x = self.downsampling_network(x)
        x = self.pre_rvq_transformer(x)
        x, loss = self.rvq(x)
        x = self.output_layer(x)
        x = torch.log_softmax(x, dim=-1)
        
        return x, loss


    def save(self, path: str):
        print("Saving model to", path)
        torch.save({"model": self.state_dict(),"options":self.options}, path)
    
    def load(self, path: str):
        print("Loading model from", path)
        model = TranscribeModel(**torch.load(path)["options"])
        model.load_state_dict(torch.load(path)["model"])
        return model