import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.utils import helpers, configs
from src.utils import hyperparameters as hp
from src.preprocessing.nn_datasets import TransformerDataset
from src.modeling.transformer import Transformer


def train_epoch(model, loader, optimizer, device, criterion):
    model.train()
    train_loss = 0.0

    for i, (history, sample, positions, target, mask) in enumerate(loader):
        if i > 0: break
        history, sample, positions, target, mask = \
            history.to(device), sample.to(device), positions.to(device), target.to(device), mask.to(device)
        optimizer.zero_grad()

        preds = model(history, sample, positions, mask)

        loss = criterion(preds, target)
        loss.backward()
        optimizer.step()

    return model


def train_transformer_fold(fold):
    helpers.set_seed(42)
    xtrn, ytrn = helpers.load_base_features(fold, mode='train')
    xval, yval = helpers.load_base_features(fold, mode='val')
    n_questions = xtrn.content_id.max()

    # group by users
    trn_group = xtrn[['user_id', 'content_id', 'answered_correctly']].groupby('user_id').apply(
        lambda r: (r['content_id'].values, r['answered_correctly'].values)
    )
    val_group = xval[['user_id', 'content_id', 'answered_correctly']].groupby('user_id').apply(
        lambda r: (r['content_id'].values, r['answered_correctly'].values)
    )
    del xtrn, ytrn, xval, yval

    # prepare sequence data
    train_set = TransformerDataset(trn_group)
    val_set = TransformerDataset(val_group)

    train_loader = DataLoader(train_set, batch_size=hp.batch_size, shuffle=False)
    val_loader = DataLoader(val_set, batch_size=hp.val_batch_size, shuffle=False)

    criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Transformer(n_questions, hp.max_seq).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)

    checkpoint_path = configs.model_dir / 'transformer'
    best_loss = {'train': np.inf, 'val': np.inf}
    log_dir = configs.log_dir / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(hp.nepochs):
        print(f"Train epoch {epoch}")
        epoch_path = checkpoint_path / f'fold{fold}_epoch{epoch}'
        model, loss = train_epoch(model, train_loader, optimizer, device, criterion)

        writer.add_scalar("Train loss", loss, epoch)
        print(f"Train loss: {loss}")

    writer.flush()
    writer.close()
