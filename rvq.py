import torch
import torch.nn as nn

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        
        # Initialize embedding with uniform distribution
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        self.commitment_cost = commitment_cost
    
    def forward(self, x):
        # Shape: (B, T, D)
        batch_size, sequence_length, embedding_dim = x.shape
        flat_x = x.reshape(batch_size * sequence_length, embedding_dim)  # (B*T, D)
        
        # Compute distances (B*T, N)
        distances = torch.cdist(
            flat_x, self.embedding.weight, p=2
        )  # p=2 for Euclidean distance
        
        # Encoding: closest embedding
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(encoding_indices).view(
            batch_size, sequence_length, embedding_dim
        )
        
        e_latent_loss = torch.mean((quantized.detach() - x) ** 2)
        q_latent_loss = torch.mean((quantized - x.detach()) ** 2)
        
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        quantized = x + (quantized - x).detach()
        
        return quantized, loss 

class ResidualVectorQuantizer(nn.Module):
    def __init__(self, num_codebooks, codebook_size, embedding_dim):
        super().__init__()
        self.codebooks = nn.ModuleList(
            [
                VectorQuantizer(codebook_size, embedding_dim)
                for _ in range(num_codebooks)
            ]
        )
    
    def forward(self, x):
        out = torch.zeros_like(x)
        total_loss = 0
        residual = x
        
        for codebook in self.codebooks:
            this_output, this_loss = codebook(residual)  # Use residual, not x
            residual = residual - this_output  # Update residual
            out = out + this_output
            total_loss += this_loss
            
        return out, total_loss
