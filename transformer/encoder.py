from torch import nn 
import torch

from embeddings import TransformerEmbedding
from model_layers import DecoderLayer, EncoderLayer 


class Encoder(nn.Module):
    def __init__(self, max_len=1000, enc_voc_size=10000, d_model=256, ffn_hidden=1024, n_heads=6, drop_prob=0.1, encoder_layers = 10, device="cpu"):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        max_len=max_len,
                                        vocab_size=enc_voc_size,
                                        drop_prob=drop_prob,
                                        device=device)

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, ffn_hidden=d_model, n_heads=n_heads, drop_pro=drop_prob) for _ in range(encoder_layers)
            ])
        
    def forward(self, x):
        x = self.emb(x)
        for layer in self.encoder_layers:
            x = layer(x)

        return x 


if __name__ == "__main__":
    input = torch.tensor([[i for i in range(100)] + [i for i in range(100)]])
    encoder = Encoder()
    output = encoder(input)
    print(f"output shape: ", output.shape)