import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.dense import Dense

class TransformerPointerGenerator(nn.Module):

    def __init__(self, encoder, decoder) -> None:
        super(TransformerPointerGenerator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_lm_head = nn.Linear(self.decoder.config.hidden_size, self.decoder.config.vocab_size, bias=False)
        self.gen_prob_proj = Dense(self.decoder.config.hidden_size, 1)

    def forward(self, encoder_input_ids, decoder_input_ids, decoder_output_ids, encoder_attention_mask=None, decoder_attention_mask=None, encoder_position_ids=None, decoder_position_ids=None, encoder_token_type_ids=None, decoder_token_type_ids=None):
        encoder_out = self.encoder(encoder_input_ids, encoder_attention_mask, encoder_position_ids, encoder_token_type_ids)
        encoder_out = encoder_out[0]
        decoder_out = self.decoder(decoder_input_ids, encoder_out, decoder_attention_mask, decoder_position_ids, decoder_token_type_ids, decoder_output_ids)
        # decoder_out: 1. decoder_output, 2. decoder_hidden_states, 3. decoder_attentions
        logits = F.softmax(self.decoder_lm_head(decoder_out[0]))
        gen_prob = self.gen_prob_proj(decoder_out[0])
        return y

    def _calc_final_dist(self, x, gen_prob, vocab_dist, attn_dists):
        vocab_dist = vocab_dist * gen_prob
        attn_dists = attn_dists * (1 - gen_prob)
        batch_size, dec_steps, ctx_len = attn_dists.size()
        dec = torch.arange(dec_steps).unsqueeze(1).unsqueeze(0).repeat(batch_size, 1, ctx_len)
        x = x.unsqueeze(1).repeat(1, dec_steps, 1)
        x = torch.stack([dec, x], dim=3)

        