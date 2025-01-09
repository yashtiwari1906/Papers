from torch import nn 
import torch

from embeddings import TransformerEmbedding
from model_layers import DecoderLayer, EncoderLayer 


class Decoder(nn.Module):
    def __init__(self, max_len=1000, enc_voc_size=10000, d_model=256, ffn_hidden=1024, n_heads=6, drop_prob=0.1, decoder_layers = 10, device="cpu"):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model=d_model, ffn_hidden=d_model, n_heads=n_heads, drop_prob=drop_prob) for _ in range(decoder_layers)
            ])
        
    def forward(self, x, target):
        x = self.emb(x)
        for layer in self.decoder_layers:
            x = layer(x, target)

        return x 

