import torch
import torch.nn as nn
from downsampling import DownsamplingNetwork
from rvq import ResidualVectorQuantizer
from self_attention import Transformer
from dataset import get_tokenizer


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
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.downsampling_network(x)  # Downsample input
        x = self.pre_rvq_transformer(x)  # Apply transformer layers
        x, loss = self.rvq(x)  # Residual vector quantization
        x = self.output_layer(x)  # Final output projection
        x = torch.log_softmax(x, dim=-1)  # Apply log softmax
        return x, loss


    def save(self, path: str):
        print("Saving model to", path)
        torch.save({
            "model_state_dict": self.state_dict(),  # Zmień z "model" na "model_state_dict"
            "options": self.options
        }, path)
    
    @classmethod
    def load(cls, path: str):  # ✅ Teraz to jest metoda klasowa
        print("Loading model from", path)
        checkpoint = torch.load(path, map_location='cpu')
        
        # Sprawdź różne formaty zapisu
        if 'options' in checkpoint:
            options = checkpoint['options']
            model = cls(**options)
            
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Fallback dla starszych formatów
            print("Warning: Loading checkpoint without options, using default parameters")
            tokenizer = get_tokenizer()  # Musisz zaimportować get_tokenizer
            model = cls(
                num_codebooks=2,  # Reduced from 4
                codebook_size=32,  # Reduced from 64
                embedding_dim=128,  # Reduced from 256
                num_transformer_layers=3,  # Reduced from 6
                vocab_size=len(tokenizer.get_vocab()),
                strides=[4, 4, 2],  # Less aggressive downsampling
                initial_mean_pooling_kernel_size=2,
                max_seq_length=200,  # Reduced from 400
            )
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        return model