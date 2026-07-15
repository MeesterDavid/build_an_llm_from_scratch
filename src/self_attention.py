import torch

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
        b, num_tokens, d_in         = x.shape
        keys                        = self.W_key(x)
        queries                     = self.W_query(x) 
        values                      = self.W_value(x)
        attention_scores            = queries @ keys.transpose(1,2)
        attention_scores.masked_fill_(self.mask.bool(), -torch.inf)
        attention_weights           = torch.softmax(attention_scores/keys.shape[-1]**0.5, dim=-1)
        attention_weights           = self.dropout(attention_weights)
        context_vectors             = attention_weights @ values
        return context_vectors
    
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
