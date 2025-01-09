import math
import torch 
from torch import nn 


class SingleHeadAttention(nn.Module):

    def __init__(self, d_model, mask = None):
        super(SingleHeadAttention, self).__init__()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.mask = mask
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        # 1. dot product with weight matrices
        batch_size, token_size, d_model = q.size() 
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        batch_size, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(1, 2)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if self.mask:
            mask = torch.triu(torch.ones(token_size, token_size), diagonal=1) == 1
            score = score.masked_fill(mask, float("-inf"))

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v

   
class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, n_heads,  mask = None):
        super(MultiHeadAttention, self).__init__()
        self.attention_blocks = [SingleHeadAttention(d_model, mask) for _ in range(n_heads)]
        self.projection = nn.Linear(d_model * n_heads, d_model)

  def forward(self, x):
    stack = [attention_block(x, x, x) for attention_block in self.attention_blocks]
    return self.projection(torch.cat(stack, dim = -1))
  


class EncoderDecoderAttention(MultiHeadAttention):
    def forward(self, query, encoded_representation_key, encoded_representation_value):
        stack = [attention_block(query, encoded_representation_key, encoded_representation_value) for attention_block in self.attention_blocks]
        return self.projection(torch.cat(stack, dim = -1)) 
