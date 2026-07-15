import torch
from torch import nn


class SelfAttentionV1(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.W_query = torch.nn.Parameter(torch.rand(input_dim, output_dim))
        self.W_key = torch.nn.Parameter(torch.rand(input_dim, output_dim))
        self.W_value = torch.nn.Parameter(torch.rand(input_dim, output_dim))
    
    def forward(self, x):
        print(f"x shape: {x.shape}")
        print(f"W_key shape: {self.W_key.shape}")
        keys                = x @ self.W_key
        queries             = x @ self.W_query 
        values              = x @ self.W_value
        attention_scores    = queries @ keys.T
        attention_weights   = torch.softmax(attention_scores/keys.shape[-1]**0.5, dim=-1)
        context_vectors     = attention_weights @ values
        return context_vectors
    
class SelfAttentionV2(torch.nn.Module):
    def __init__(self, input_dim, output_dim, qkv_bias=False):
        super().__init__()
        self.W_query    = torch.nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_key      = torch.nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_value    = torch.nn.Linear(input_dim, output_dim, bias=qkv_bias)
    
    def forward(self, x):
        keys                = self.W_key(x)
        queries             = self.W_query(x) 
        values              = self.W_value(x)
        attention_scores    = queries @ keys.T
        attention_weights   = torch.softmax(attention_scores/keys.shape[-1]**0.5, dim=-1)
        context_vectors     = attention_weights @ values
        return context_vectors
    
class CausalAttention(torch.nn.Module):
    def __init__(self, input_dim, output_dim, context_length, dropout=0.5,qkv_bias=False):
        super().__init__()
        self.W_query    = torch.nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_key      = torch.nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.W_value    = torch.nn.Linear(input_dim, output_dim, bias=qkv_bias)
        self.dropout    = torch.nn.Dropout(dropout)
        
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
    
    def forward(self, x):
        keys                        = self.W_key(x)
        queries                     = self.W_query(x) 
        values                      = self.W_value(x)
        attention_scores            = queries @ keys.transpose(1,2)
        attention_scores.masked_fill_(self.mask.bool(), -torch.inf)
        attention_weights           = torch.softmax(attention_scores/keys.shape[-1]**0.5, dim=-1)
        attention_weights           = self.dropout(attention_weights)
        context_vectors             = attention_weights @ values
        return context_vectors
    
class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)
        ])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # optional projection

        return context_vec


inputs = torch.tensor(
    [[0.43, 0.15, 0.89],
    [0.55, 0.87, 0.66],
    [0.57, 0.85, 0.64],
    [0.22, 0.58, 0.33],
    [0.77, 0.25, 0.10],
    [0.05, 0.80, 0.55]]
)

torch.manual_seed(123)
sa_v1 = CausalAttention(3, 2, 6, 0.5)
batch = torch.stack((inputs, inputs), dim=0)
print(sa_v1(batch))
