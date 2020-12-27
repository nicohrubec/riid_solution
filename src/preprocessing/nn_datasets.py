import numpy as np
from torch.utils.data import Dataset


class TransformerDataset(Dataset):
    def __init__(self, group, max_seq=100, num_feats=2):
        super(TransformerDataset, self).__init__()
        self.max_seq = max_seq
        self.samples = {}
        self.sample_ids = []
        self.group = group
        self.num_feats = num_feats

        for user in self.group.index:
            # get features for user
            user_q, user_a = self.group[user]
            user_seq_len = len(user_q)
            user_pos = [pos for pos in range(user_seq_len)]
            user_seq = np.zeros((self.num_feats + 1, user_seq_len), dtype=np.float32)

            # assign feats to user array and master user list
            user_seq[0] = user_a
            user_seq[1] = user_q
            user_seq[2] = user_pos
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
        sample_history = np.zeros((self.num_feats + 1, self.max_seq), dtype=np.float32)
        sample_history[:, :row_id] = sample_data
        sample = self.samples[user][1:, row_id]
        target = self.samples[user][0, row_id]

        # mask not filled entries
        mask = np.zeros(self.max_seq, dtype=np.int8)
        mask[:row_id] = 1

        return sample_history, sample, target, mask
