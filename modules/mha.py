import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_attention_heads) -> None:
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_attention_heads == 0
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        batch_size, seq_length, hidden_size = k.size()
        d_attention_head = int(hidden_size / self.num_attention_heads)
        k = k.view(batch_size, seq_length, self.num_attention_heads, d_attention_head).transpose(1,2)
        v = v.view(batch_size, seq_length, self.num_attention_heads, d_attention_head).transpose(1,2)

        batch_size, seq_length, hidden_size = q.size()
        d_attention_head = int(hidden_size / self.num_attention_heads)
        q = q.view(batch_size, seq_length, self.num_attention_heads, d_attention_head).transpose(1,2)

        qk_T = torch.matmul(q, k.transpose(2,3))
        qk_T = qk_T / d_attention_head ** 0.5
        alpha = F.softmax(qk_T, dim=-1)
        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(alpha)
            alpha = alpha.masked_fill(mask, -1e9)
        hidden_states = torch.matmul(alpha, v)
        hidden_states = hidden_states.transpose(1,2).contiguous().view(batch_size, seq_length, hidden_size)
        return hidden_states, alpha