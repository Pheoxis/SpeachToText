import math

import torch.nn.functional as F
import torch
import torch.nn as nn

def calculate_attention(
    values: torch.Tensor,
    keys: torch.Tensor,
    query: torch.Tensor,
):
    # Compute attention scores using query-key dot product
    attention_scores = torch.matmul(query, keys.transpose(-2, -1))
    
    # Scale by square root of key dimension (scaled dot-product attention)
    attention_scores = attention_scores / math.sqrt(keys.shape[-1])
    
    # Apply softmax to get attention weights
    attention_scores = F.softmax(attention_scores, dim=-1)
    
    # Apply attention weights to values
    attention = torch.matmul(attention_scores, values)
    
    return attention, attention_scores




class FeedForward(nn.Module):
    def __init__(self, embed_size: int):
        super().__init__()
        self.layer1 = nn.Linear(embed_size, embed_size)
        self.layer2 = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        x = self.layer1(x)
        x = F.gelu(x)  # GELU activation function
        x = self.layer2(x)
        return x

class SelfAttentionLayer(nn.Module):
    def __init__(self, embed_size: int):
        super().__init__()
        self.embed_size = embed_size
        
        # Linear projections for queries, keys, and values
        self.query_dense = nn.Linear(embed_size, embed_size)
        self.key_dense = nn.Linear(embed_size, embed_size)
        self.value_dense = nn.Linear(embed_size, embed_size)

    def forward(self, embeddings: torch.Tensor):
        # Project embeddings to queries, keys, and values
        query = self.query_dense(embeddings)
        key = self.key_dense(embeddings)
        value = self.value_dense(embeddings)
        
        # Calculate attention using the previously defined function
        attention, _ = calculate_attention(value, key, query)
        
        return attention


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size: int, num_heads: int):
        super().__init__()
        
        # Ensure embedding size is divisible by number of heads
        assert (
            embed_size % num_heads == 0
        ), "Embedding size must be divisible by number of heads"
        
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        
        # Linear projection layers (these were missing from your __init__)
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        self.output_linear = nn.Linear(embed_size, embed_size)  # Fixed typo: "otput_liear"
    
    def forward(self, embeddings: torch.Tensor):
        batch_size = embeddings.shape[0]
        seq_length = embeddings.shape[1]
        
        # Linear projections and reshape to separate heads
        query = self.query(embeddings).reshape(
            batch_size, seq_length, self.num_heads, self.head_dim
        )
        key = self.key(embeddings).reshape(
            batch_size, seq_length, self.num_heads, self.head_dim
        )
        value = self.value(embeddings).reshape(
            batch_size, seq_length, self.num_heads, self.head_dim
        )
        
        # Transpose for efficient attention computation
        query = query.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        key = key.transpose(1, 2)      # [batch, num_heads, seq_len, head_dim]
        value = value.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        
        # Compute scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.head_dim
        )
        
        # Apply softmax to convert scores to probabilities
        attention_scores = F.softmax(attention_scores, dim=-1)
        
        # Apply attention weights to values
        attention = torch.matmul(attention_scores, value)
        
        # Concatenate heads back together (this was the missing part)
        attention = attention.transpose(1, 2).contiguous().reshape(
            batch_size, seq_length, self.embed_size
        )
        
        # Final output projection
        output = self.output_linear(attention)
        
        return output
        
class TransformerBlock(nn.Module):
    def __init__(self, embed_size: int):
        super().__init__()
        self.attention_layer = SelfAttentionLayer(embed_size)
        self.feed_forward = FeedForward(embed_size)
        self.layer_norm1 = nn.LayerNorm(embed_size)
        # self.layer_norm2 = nn.LayerNorm(embed_size)  # ADD THIS
        # self.dropout = nn.Dropout(0.1)
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
        
        # Create position indices
        position = torch.arange(max_seq_length).unsqueeze(1)
        
        # Calculate division term for frequency scaling
        div_term = torch.exp(
            torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size)
        )
        
        # Initialize positional encoding matrix
        pe = torch.zeros(max_seq_length, embed_size)
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices  
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of model state)
        self.register_buffer("positional_embedding", pe)

    def forward(self, x: torch.Tensor):
        return x + self.positional_embedding[: x.size(1), :]
    
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
