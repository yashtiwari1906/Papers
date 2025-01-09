
from torch import nn
import torch

from decoder import Decoder
from encoder import Encoder 

class Transformer(nn.Module):
    def __init__(self,  max_len=1000, enc_voc_size=10000, dec_voc_size = 20000, d_model=256, ffn_hidden=1024, n_head=6, drop_prob=0.1, device = "cpu"):
        super().__init__()  
        # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device 
        self.encoder = Encoder(max_len=max_len, enc_voc_size=enc_voc_size, d_model=d_model, ffn_hidden=ffn_hidden, n_heads=n_head, drop_prob=drop_prob, device=device)
        self.decoder = Decoder(max_len=max_len, enc_voc_size=enc_voc_size, d_model=d_model, ffn_hidden=ffn_hidden, n_heads=n_head, drop_prob=drop_prob, device=device)
        self.linear = nn.Linear(d_model, dec_voc_size) 
        self.softmax = nn.Softmax(dim=-1) 

    def forward(self, src, target):
        encoded_src = self.encoder(src)
        output = self.decoder(target, encoded_src)
        output = self.linear(output)
        output = self.softmax(output)
        return output 
    




if __name__ == "__main__":
    device = "cpu" #torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = Transformer(device = device)
    input = torch.rand(1, 3, 256)

    src =  torch.tensor([[i for i in range(10)]])
    target = torch.tensor([[i for i in range(15)]])
    output = model(src, target=target)
    print("*"*50)
    print(output[-1][-1].shape)
    print("*"*50)

    # idx = torch.argmax(output[-1][-1], dim = 1)
    print("="*50)
    print(output.shape)
    print(output)
    # print(f"idx: {idx}")
    