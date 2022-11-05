import torch.nn as nn
from modules.mha import MultiHeadAttention
from modules.ffn import FeedForwardNetwork

class EncoderLayer(nn.Module):
    
    def __init__(self, d_model, num_attention_heads, d_ff, drop_rate=0.1) -> None:
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, num_attention_heads, drop_rate=0.1)
        self.ln1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_rate)
        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_rate)

    def forward(self, x, mask=None):
        (attn_out, )  = self.attention(x,x,x,mask)
        x = self.ln1(attn_out + x)
        x = self.dropout1(x)
        
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        x = self.dropout2(x)
        return x

if __name__ == '__main__':
    encoder_layer = EncoderLayer()