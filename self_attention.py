import math

import torch.nn.functional as F
import torch
import torch.nn as nn

def calculate_attention(values, keys, query, chunk_size=256):
    """Memory-efficient chunked attention."""
    batch_size, seq_len, embed_dim = query.shape
    
    if seq_len <= chunk_size:
        # Normal attention for short sequences
        attention_scores = torch.matmul(query, keys.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(keys.shape[-1])
        attention_scores = torch.softmax(attention_scores, dim=-1)
        attention = torch.matmul(attention_scores, values)
        return attention, attention_scores
    
    # Chunked attention for long sequences
    output = torch.zeros_like(query)
    
    for i in range(0, seq_len, chunk_size):
        end_i = min(i + chunk_size, seq_len)
        query_chunk = query[:, i:end_i, :]
        
        attention_scores = torch.matmul(query_chunk, keys.transpose(-2, -1))
        attention_scores = attention_scores / math.sqrt(keys.shape[-1])
        attention_scores = torch.softmax(attention_scores, dim=-1)
        
        attention_chunk = torch.matmul(attention_scores, values)
        output[:, i:end_i, :] = attention_chunk
    
    return output, None



class FeedForward(nn.Module):
    def __init__(self, embed_size: int):
        super().__init__()
        self.layer1 = nn.Linear(embed_size, embed_size)
        self.layer2 = nn.Linear(embed_size, embed_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)
        x = self.layer2(x)
        return x
class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_size: int):
        super().__init__()
        self.embed_size = embed_size
        self.query_dense = nn.Linear(embed_size, embed_size)
        self.key_dense = nn.Linear(embed_size, embed_size)
        self.value_dense = nn.Linear(embed_size, embed_size)
    
    def forward(self, embeddings: torch.Tensor):
        query = self.query_dense(embeddings)
        key = self.key_dense(embeddings)
        value = self.value_dense(embeddings)
        attention, _ = calculate_attention(value, key, query)
        return attention

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size: int, num_heads: int):
        super().__init__()
        assert (
            embed_size % num_heads == 0
        ), "Embedding size must be divisible by number of heads"
        
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        # Single linear layer for each of Q, K, V that will be split into heads
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
        self.attention = SelfAttentionLayer(embed_size)
        
class TransformerBlock(nn.Module):
    def __init__(self, embed_size: int):
        super().__init__()
        self.attention_layer = SelfAttentionLayer(embed_size)
        self.feed_forward = FeedForward(embed_size)
        self.layer_norm1 = nn.LayerNorm(embed_size)
        
    def forward(self, x: torch.Tensor):
        context = self.attention_layer(x)
        context = self.layer_norm1(context)
        context = self.feed_forward(context)
        context = F.gelu(context)
        output = context + x
        return output

class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, embed_size: int, max_seq_length: int):
        super().__init__()
        self.embed_size = embed_size
        self.max_seq_length = max_seq_length
        
        # Create initial positional embeddings
        pe = self._create_positional_encoding(max_seq_length, embed_size)
        self.register_buffer("positional_embedding", pe)

    def _create_positional_encoding(self, seq_length: int, embed_size: int):
        """Create positional encoding for given sequence length."""
        position = torch.arange(seq_length).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size)
        )
        
        pe = torch.zeros(seq_length, embed_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: torch.Tensor):
        seq_len = x.size(1)
        
        # If input sequence is longer than pre-computed embeddings, extend them
        if seq_len > self.positional_embedding.size(0):
            # Create new positional encoding for the required length
            extended_pe = self._create_positional_encoding(seq_len, self.embed_size)
            extended_pe = extended_pe.to(self.positional_embedding.device)
            
            # Update the buffer with the extended version
            self.register_buffer("positional_embedding", extended_pe)
        
        return x + self.positional_embedding[:seq_len, :]


class Transformer(nn.Module):
    def __init__(self, embed_size: int, num_layers: int, max_seq_length: int):
        super().__init__()
        self.positional_encoding = SinusoidalPositionEncoding(
            embed_size, max_seq_length
        )
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(embed_size) for _ in range(num_layers)]
        )
    
    def forward(self, x: torch.Tensor):
        x = self.positional_encoding(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        return x
