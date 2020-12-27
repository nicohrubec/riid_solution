import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer

from src.utils import hyperparameters as hp


class Transformer(nn.Module):
    def __init__(self, n_questions, max_seq):
        super(Transformer, self).__init__()
        # embed user history
        self.q_hist_emb = nn.Embedding(n_questions + 1, hp.embed_size)
        self.pos_hist_emb = nn.Embedding(max_seq + 1, hp.embed_size * 2)
        self.a_hist_emb = nn.Linear(1, hp.embed_size)

        # embed current sample for user
        self.q_sample_emb = nn.Embedding(n_questions + 1, hp.embed_size * 2)
        self.pos_sample_emb = nn.Embedding(max_seq + 1, hp.embed_size)
        self.sample_emb = nn.Linear(hp.embed_size, hp.embed_size * 2)

        # transformer
        self.encoder_layer = TransformerEncoderLayer(
            d_model=hp.embed_size * 2,
            nhead=hp.num_heads,
            dropout=hp.dropout,
            dim_feedforward=hp.dim_ff
        )
        self.transformer = TransformerEncoder(self.encoder_layer, hp.num_enc_layers)
        self.avgpool = nn.AvgPool1d(kernel_size=max_seq)

        self.out1 = nn.Linear(hp.embed_size * 4, hp.embed_size * 4)
        self.out2 = nn.Linear(hp.embed_size * 4, 1)

    def forward(self, history, curr_sample, pos, mask):
        # history input
        a_history = self.a_hist_emb(torch.unsqueeze(history[:, 0], 2))
        q_history = self.q_hist_emb(history[:, 1].long())
        pos_history = self.pos_hist_emb(pos)

        # assemble history representation
        history = torch.cat((a_history, q_history), dim=2)
        history = torch.add(history, pos_history)
        history = history.permute(1, 0, 2)

        history = self.transformer(history, src_key_padding_mask=mask)
        history = history.permute(1, 2, 0)
        history = self.avgpool(history).squeeze()

        # sample input
        q_sample = self.q_sample_emb(curr_sample[:, 0].long())
        # get position embedding?

        # head
        out = torch.cat((history, q_sample), axis=1)
        out = self.out1(out)
        out = self.out2(out)

        return out.squeeze()
