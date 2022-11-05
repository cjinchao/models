import torch
import torch.nn as nn
from modules.mha import MultiHeadAttention
from .ffn import FeedForwardNetwork

from transformers import GPT2Model

class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model=512, n_heads=8, d_ff=2048, dropout=0.1, layer_norm_eps=1e-6, max_position=512) -> None:
        super(TransformerDecoderLayer, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        
        self.s_attn = MultiHeadAttention(d_model, n_heads)
        self.c_attn = MultiHeadAttention(d_model, n_heads)

        self.ff = FeedForwardNetwork(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)

        self.causal_mask = self._get_subsequent_mask(max_position)
        self.register_buffer('mask', self.causal_mask)

    def forward(self, inputs, memory, src_pad_mask, tgt_pad_mask, previous_inputs=None):
        decoder_mask = torch.gt(tgt_pad_mask + self.causal_mask[:, :tgt_pad_mask.size(1), :tgt_pad_mask.size(1)], 0)

        inputs = self.norm1(inputs)
        all_input = inputs
        if previous_inputs is not None:
            all_input = torch.cat((previous_inputs, inputs), dim=1)
            decoder_mask = None
        q, _ = self.s_attn(all_input, all_input, all_input, decoder_mask)
        q = self.dropout1(q) + inputs
        q_norm = self.norm2(q)
        ctx, _ = self.c_attn(q_norm, memory, memory, mask=src_pad_mask)
        output = self.ff(self.dropout1(ctx) + q)
        return output, all_input

    def _get_subsequent_mask(self, sz):
        attn_shape = (1, sz, sz)
        return torch.triu(torch.ones(attn_shape))

class TransformerDecoder(nn.Module):

    def __init__(self, n_layers, d_model, n_heads, d_ff, dropout, embeddings) -> None:
        super(TransformerDecoder, self).__init__()
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.embeddings = embeddings
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout) for _ in range(n_layers)])
        # self.pos_emb = PositionalEncoding(d_model, dropout=dropout)

    def forward(self, input_ids, encoder_hidden_states, encoder_input_ids, previous_inputs=None):
        (src_batch_size, src_seq_length, *_) = encoder_hidden_states.size()
        (tgt_batch_size, tgt_seq_length, *_) = input_ids.size()
        
        output = self.embeddings(input_ids)
        # assert output.dim() == 3
        # output = self.pos_emb(output)

        padding_idx = self.embeddings.padding_idx
        tgt_pad_mask = input_ids.eq(padding_idx).unsqueeze(1).expand(tgt_batch_size, tgt_seq_length, tgt_seq_length)
        src_pad_mask = encoder_input_ids.eq(padding_idx).unsqueeze(1).expand(src_batch_size, tgt_seq_length, src_seq_length)
        # TODO src and tgt padding masks handled here
        for i in range(self.n_layers):
            output, all_input = self.decoder_layers[i](output, encoder_hidden_states, src_pad_mask, tgt_pad_mask)
        return output, all_input

if __name__ == '__main__':
    sz = 8
    attn_shape = (1, sz, sz)
    print(torch.triu(torch.ones(attn_shape)))