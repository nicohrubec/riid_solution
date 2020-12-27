import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import DataLoader

from src.utils import helpers, configs
from src.preprocessing.nn_datasets import TransformerDataset


def train_transformer_fold(fold):
    helpers.set_seed(42)
    xtrn, ytrn = helpers.load_base_features(fold, mode='train')
    xval, yval = helpers.load_base_features(fold, mode='val')

    trn_group = xtrn[['user_id', 'content_id', 'answered_correctly']].groupby('user_id').apply(
        lambda r: (r['content_id'].values, r['answered_correctly'].values)
    )
    val_group = xval[['user_id', 'content_id', 'answered_correctly']].groupby('user_id').apply(
        lambda r: (r['content_id'].values, r['answered_correctly'].values)
    )
    del xtrn, ytrn, xval, yval

    # prepare sequence data
    train_set = TransformerDataset(trn_group)
    print(train_set[40])
