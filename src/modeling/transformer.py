import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

from src.utils import hyperparameters as hp


class Transformer(nn.Module):
    def __init__(self, n_questions, max_seq, n_parts, max_time_lag):
        super(Transformer, self).__init__()
        # embed user history
        self.q_hist_emb = nn.Embedding(n_questions + 1, hp.embed_size)
        self.p_hist_emb = nn.Embedding(n_parts + 1, hp.embed_size)
        self.t_hist_emb = nn.Embedding(max_time_lag + 1, hp.embed_size)
        self.a_hist_emb = nn.Linear(1, hp.embed_size)
        self.pos_hist_emb = nn.Embedding(max_seq + 1, hp.embed_size * 4)

        # embed current sample for user
        self.q_sample_emb = nn.Embedding(n_questions + 1, hp.embed_size * 2)
        self.p_sample_emb = nn.Embedding(n_parts + 1, hp.embed_size)
        self.t_sample_emb = nn.Embedding(max_time_lag + 1, hp.embed_size)
        self.sample_emb = nn.Linear(hp.embed_size * 4, hp.embed_size * 4)

        # transformer
        self.encoder_layer = TransformerEncoderLayer(
            d_model=hp.embed_size * 4,
            nhead=hp.num_heads,
            dropout=hp.dropout,
            dim_feedforward=hp.dim_ff
        )
        self.transformer = TransformerEncoder(self.encoder_layer, hp.num_enc_layers)
        self.avgpool = nn.AvgPool1d(kernel_size=max_seq)

        self.out1 = nn.Linear(hp.embed_size * 8, hp.embed_size * 8)
        self.out2 = nn.Linear(hp.embed_size * 8, 1)

    def forward(self, history, curr_sample, pos, mask):
        # history input
        a_history = self.a_hist_emb(torch.unsqueeze(history[:, 0], 2).float())
        q_history = self.q_hist_emb(history[:, 1])
        p_history = self.p_hist_emb(history[:, 2])
        t_history = self.t_hist_emb(history[:, 3])
        pos_history = self.pos_hist_emb(pos)

        # assemble history representation
        history = torch.cat((a_history, q_history, p_history, t_history), dim=2)
        history = torch.add(history, pos_history)
        history = history.permute(1, 0, 2)

        history = self.transformer(history)
        history = history.permute(1, 2, 0)
        history = self.avgpool(history).squeeze()

        # sample input
        q_sample = self.q_sample_emb(curr_sample[:, 0])
        p_sample = self.p_sample_emb(curr_sample[:, 1])
        t_sample = self.t_sample_emb(curr_sample[:, 2])
        sample_emb = self.sample_emb(torch.cat((q_sample, p_sample, t_sample), dim=1))
        # get position embedding?

        # head
        out = torch.cat((history, sample_emb), dim=1)
        out = self.out1(out)
        out = self.out2(out)

        return out.squeeze()
