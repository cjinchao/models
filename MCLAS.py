import torch
import torch.nn as nn

from transformers import XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer

from modules.decoder import TransformerDecoder
        
class MCLAS(nn.Module):

    def __init__(self, args) -> None:
        super(MCLAS, self).__init__()
        self.args = args
        config = XLMRobertaConfig(vocab_size=args.vocab_size, max_position_embeddings=args.max_position_embeddings, num_attention_heads=args.n_heads, num_hidden_layers=args.n_layers, hidden_size=args.d_model, intermediate_size=args.d_ff, hidden_dropout_prob=args.dropout, attention_probs_dropout_prob=args.dropout, layer_norm_eps=args.layer_norm_eps)
        self.encoder = XLMRobertaModel(config)
        self.vocab_size = self.encoder.config.vocab_size
        if args.shared_embeddings:
            # decoder_embeddings = copy.deepcopy(self.encoder.embeddings.word_embeddings.weight)
            decoder_embeddings = self.encoder.embeddings
        else:
            decoder_embeddings = nn.Embedding(self.vocab_size, self.encoder.config.hidden_size)

        self.decoder = TransformerDecoder(n_layers=args.n_layers, d_model=args.d_model, n_heads=args.n_heads, d_ff=args.d_ff, dropout=args.dropout, embeddings=decoder_embeddings)
        self.decoder.embeddings = decoder_embeddings
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        encoder_out = self.encoder(input_ids, token_type_ids, attention_mask)
        encoder_out = encoder_out[0]
        decoder_output, _ = self.decoder(labels, encoder_out, input_ids)
        return decoder_output

if __name__ == '__main__':
    
    class Args(object):
        
        def __init__(self, **kwargs) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    
    args = Args(bert_model = 'xlm-roberta-base', d_model=512, n_layers=6, n_heads=8, d_ff=2048, dropout=0.1, shared_embeddings=True, max_position_embeddings=1024, layer_norm_eps=1e-6)
    tokenier = XLMRobertaTokenizer.from_pretrained(args.bert_model)
    setattr(args, 'vocab_size', tokenier.vocab_size)
    mclas = MCLAS(args)

    with open('/Users/caijinchao/Desktop/bert_data_enenzh_seg/ncls.test.0.bert.pt', 'rb') as f:
        data = torch.load(f)
    data = data[0]
    src = " ".join(data['src_txt'])
    tgt = " ".join(data['tgt_txt_eng'])
    tgt_zh = " ".join([data['tgt_txt']])
    src = tokenier.encode(src, return_tensors='pt', add_special_tokens=False)
    tgt = tokenier.encode(tgt, return_tensors='pt', add_special_tokens=False)
    tgt_zh = tokenier.encode(tgt_zh, return_tensors='pt', add_special_tokens=False)
    tgt = torch.concat([tgt, tgt_zh], dim=1)
    out = mclas(src, labels=tgt)