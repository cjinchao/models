from json import decoder
import torch.nn as nn
from transformers import XLMRobertaConfig, XLMRobertaModel, GPT2Model, GPT2Config

class VAE(nn.Module):

    def __init__(self, encoder_config, decoder_config) -> None:
        super(VAE, self).__init__()
        self.encoder_config = XLMRobertaConfig.from_pretrained(encoder_config['encoder_model'])
        self.encoder = XLMRobertaModel.from_pretrained(encoder_config['encoder_model'])
        
        self.decoder_config = GPT2Config.from_pretrained(decoder_config['decoder_model'])
        self.decoder = GPT2Model.from_pretrained(decoder_config['decoder_model'])
        self.decoder.resize_token_embeddings(self.encoder.config.vocab_size)
        self.decoder.wte = self.encoder.embeddings.word_embeddings
        self.decoder.wpe = self.encoder.embeddings.position_embeddings

        self.mean_proj = nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size)
        self.logvar_proj = nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        encoder_out = self.encoder(input_ids, token_type_ids, attention_mask)
        encoder_out = encoder_out[0]
        decoder_output, _ = self.decoder(labels, encoder_out, input_ids)
        return decoder_output

if __name__ == "__main__":
    encoder_config = {'encoder_model': 'xlm-roberta-base'}
    decoder_config = {'decoder_model': 'gpt2'}
    vae = VAE(encoder_config, decoder_config)
    pass