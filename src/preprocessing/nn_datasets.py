import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import hyperparameters as hp


class TransformerDataset(Dataset):
    def __init__(self, group, max_seq=hp.max_seq, num_feats=4):
        super(TransformerDataset, self).__init__()
        self.max_seq = max_seq
        self.samples = {}
        self.sample_ids = []
        self.group = group
        self.num_feats = num_feats
        self.max_time_lag = 0

        for user in self.group.index:
            # get features for user
            user_q, user_a, user_p, user_t = self.group[user]
            user_seq_len = len(user_q)
            user_seq = np.zeros((self.num_feats, user_seq_len), dtype=np.float32)

            # assign feats to user array and master user list
            user_seq[0] = user_a
            user_seq[1] = user_q
            user_seq[2] = user_p
            user_seq[3] = np.clip(np.diff(user_t, prepend=0), a_max=10080, a_min=0)  # time lag
            self.max_time_lag = max(np.max(user_seq[3]), self.max_time_lag)
            self.samples[user] = user_seq

            # get master user id list for sampling
            user_ids = [str(user) + '_' + str(pos) for pos in range(user_seq_len)]
            self.sample_ids.extend(user_ids)

    # this is called to reset the master sampling list after each epoch
    def reset_id_list(self):
        # empty sample ids
        self.sample_ids = []

        # add id for each user for each row to master id list
        for user in self.group.index:
            user_q, user_a = self.group[user]
            user_seq_len = len(user_q)
            user_ids = [str(user) + '_' + str(pos) for pos in range(user_seq_len)]
            self.sample_ids.extend(user_ids)

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        # get sample id at index
        str_user_id = self.sample_ids[idx]

        # extract user and row id
        user, row_id = str_user_id.split('_')
        user, row_id = int(user), int(row_id)

        # get user history and sample to be predicted + target for row
        sample_data = self.samples[user][:, :row_id]
        sample_history = np.zeros((self.num_feats, self.max_seq), dtype=np.float32)
        sample_history[:self.num_feats, :row_id] = sample_data[:, -self.max_seq:]
        position = np.zeros(self.max_seq, dtype=np.int16)
        position[:row_id] = [pos for pos in range(min(self.max_seq, row_id), 0, -1)]
        sample = self.samples[user][1:, row_id]
        target = self.samples[user][0, row_id]

        # mask not filled entries
        mask = np.zeros(self.max_seq, dtype=np.int8)
        mask[:row_id] = 1

        sample_history = torch.from_numpy(sample_history).long()
        sample = torch.from_numpy(sample).long()
        position = torch.from_numpy(position).long()
        mask = torch.from_numpy(mask).bool()

        return sample_history, sample, position, target, mask

    def get_max_time_lag(self):
        return int(self.max_time_lag)
