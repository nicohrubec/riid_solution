import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import datetime
from tqdm import tqdm

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

    for i, (history, sample, positions, target, mask) in enumerate(tqdm(loader)):
        history, sample, positions, target, mask = \
            history.to(device), sample.to(device), positions.to(device), target.to(device), mask.to(device)
        optimizer.zero_grad()

        preds = model(history, sample, positions, mask)

        loss = criterion(preds, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() / len(loader)

    return model, train_loss


def validate(model, loader, device, criterion):
    model.eval()
    val_loss = 0.0
    val_preds = []
    val_targets = []

    with torch.no_grad():
        for i, (history, sample, positions, target, mask) in enumerate(tqdm(loader)):
            history, sample, positions, target, mask = \
                history.to(device), sample.to(device), positions.to(device), target.to(device), mask.to(device)

            preds = model(history, sample, positions, mask)
            loss = criterion(preds, target)
            val_loss += loss.item() / len(loader)

            val_preds.append(preds.cpu())
            val_targets.append(target.cpu())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_auc = roc_auc_score(val_targets, val_preds)

    return val_loss, val_auc


def train_transformer_fold(fold):
    helpers.set_seed(42)
    xtrn, ytrn = helpers.load_base_features(fold, mode='train', tail=True)
    xval, yval = helpers.load_base_features(fold, mode='val', tail=True)
    n_questions = max(xtrn.content_id.max(), xval.content_id.max())
    n_parts = max(xtrn.part.max(), xval.part.max())

    # group by users
    trn_group = xtrn[['user_id', 'content_id', 'answered_correctly', 'part']].groupby('user_id').apply(
        lambda r: (r['content_id'].values, r['answered_correctly'].values, r['part'].values)
    )
    val_group = xval[['user_id', 'content_id', 'answered_correctly', 'part']].groupby('user_id').apply(
        lambda r: (r['content_id'].values, r['answered_correctly'].values, r['part'].values)
    )
    del xtrn, ytrn, xval, yval

    # prepare sequence data
    train_set = TransformerDataset(trn_group)
    val_set = TransformerDataset(val_group)

    train_loader = DataLoader(train_set, batch_size=hp.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=hp.val_batch_size, shuffle=False, num_workers=4, pin_memory=True)

    criterion = nn.BCEWithLogitsLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Transformer(n_questions, hp.max_seq, n_parts).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)

    checkpoint_path = configs.model_dir / 'transformer'
    best_loss = {'train': np.inf, 'val': np.inf, 'val_auc': 0}
    log_dir = configs.log_dir / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(1, hp.nepochs + 1):
        print(f"Train epoch {epoch}")
        model, loss = train_epoch(model, train_loader, optimizer, device, criterion)
        val_loss, val_auc = validate(model, val_loader, device, criterion)

        # save best model
        if val_auc > best_loss['val_auc']:
            best_loss = {'train': loss, 'val': val_loss, 'val_auc': val_auc}
            epoch_path = checkpoint_path / f'fold{fold}_epoch{epoch}_{val_auc}.pt'
            torch.save(model.state_dict(), epoch_path)

        # log losses
        writer.add_scalar("Train loss", loss, epoch)
        writer.add_scalar("Val loss", val_loss, epoch)
        writer.add_scalar("Val auc", val_auc, epoch)

        print("Train loss: {:5.5f}".format(loss))
        print("Val loss: {:5.5f}    Val auc: {:4.4f}".format(val_loss, val_auc))

    writer.flush()
    writer.close()
