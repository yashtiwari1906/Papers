import torch.nn as nn
import torch
from torchvision import models
from collections import OrderedDict
from torchvision import models

from attention import EncoderDecoderAttention, MultiHeadAttention
from layers import LayerNorm, PositionwiseFeedForward
#from torchsummary import summary

class EncoderLayer(nn.Module):
    def __init__(self,  d_model=256, ffn_hidden=1024, n_heads=6, drop_pro=0.1):
        super().__init__() 
        self.multiheaded_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads, mask =  False)
        self.ffnn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden)
        self.layer_norm1 = LayerNorm(d_model=d_model)
        self.layer_norm2 = LayerNorm(d_model=d_model)
        self.drop_out1 = nn.Dropout(drop_pro)
        self.drop_out2 = nn.Dropout(drop_pro)

    def forward(self, x):
        x_ = x 
        x = self.multiheaded_attention(x)
        x = self.drop_out1(x)
        x = self.layer_norm1(x_ + x)
        x_ = x 
        x = self.ffnn(x)
        x = self.drop_out2(x)
        x = self.layer_norm2(x_ + x)
        return x 

class DecoderLayer(nn.Module):
    def __init__(self, d_model=256, ffn_hidden=1024, n_heads=6, drop_prob=0.1):
        super().__init__()
        self.masked_self_attention = MultiHeadAttention(d_model=d_model, n_heads=n_heads, mask=True)
        self.layer_norm1 = LayerNorm(d_model=d_model)
        self.drop_out1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = EncoderDecoderAttention(d_model=d_model, n_heads=n_heads)
        self.layer_norm2 = LayerNorm(d_model=d_model)
        self.drop_out2 = nn.Dropout(p=drop_prob)

        self.ffnn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob)
        self.layer_norm3 = LayerNorm(d_model=d_model)
        self.drop_out3 = nn.Dropout(p=drop_prob)

    def forward(self, encoded_tensors, x):
        x_ = x 
        x = self.masked_self_attention(x)
        x = self.drop_out1(x)
        x = self.layer_norm1(x_ + x)
        x_ = x 
        x = self.enc_dec_attention(query = x, encoded_representation_key=encoded_tensors, encoded_representation_value=encoded_tensors)
        x = self.drop_out2(x)
        x = self.layer_norm2(x_ + x)
        x_ = x 
        x = self.ffnn(x)
        x = self.layer_norm3(x_ + x)
        return x 



        

if __name__ == "__main__":
    encoder = EncoderLayer()
    input = torch.rand(1, 3, 256)
    output = encoder(input)
    print("="* 30)
    print("ENCODER")
    print("="* 30)
    print(output)
    print(output.shape)
    encoded_attns = torch.rand(1, 3, 256)
    decoder = DecoderLayer() 
    output = decoder(encoded_attns, input)
    print("="* 30)
    print("DECODER")
    print("="* 30)
    print(output)
    print(output.shape)
    











