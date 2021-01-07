import pickle

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import datetime
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.preprocessing import feature_engineering, nn_dataset
from src.modeling import nn_model
from src.utils import configs, helpers
from src.utils import hyperparameters as hp


def train_fold(fold, model_type):
    assert model_type in ['cat', 'lgb', 'nn']
    # get base features
    feats = ['content_id', 'task_container_id', 'prior_question_elapsed_time', 'prior_question_had_explanation', 'part',
             'content_id_target_mean', 'user_count', 'user_correct_mean', 'user_question_count',
             'user_question_correct_mean', 'last_time_user', 'last_time_question_user', 'last_time_user_inter',
             'user_last_n_correct', 'answer1', 'answer2', 'answer3', 'answer4', 'user_last_n_time', 'user_last_n_time2',
             'user_last_n_time3', 'user_part_count', 'user_part_correct_mean', 'user_part1_mean', 'user_part2_mean',
             'user_part3_mean', 'user_part4_mean', 'user_part5_mean', 'user_part6_mean', 'user_part7_mean',
             'task_container_eq1', 'task_container_eq2',
             'user_hardness_count', 'user_hardness_mean', 'user_hardness_inter']
    target = 'answered_correctly'
    xtrn, ytrn = helpers.load_base_features(fold, mode='train')
    xval, yval = helpers.load_base_features(fold, mode='val')

    # get features
    xtrn, xval = feature_engineering.get_global_stats(xtrn, xval, target)
    xtrn, xval = feature_engineering.get_answer_feats(xtrn, xval)
    xtrn, xval = feature_engineering.get_user_feats(xtrn, xval)

    # train model on single fold
    if model_type == 'lgb':
        model = train_lgb(xtrn, ytrn, xval, yval, feats, plot=False)
    elif model_type == 'cat':
        model = train_cat(xtrn, ytrn, xval, yval, feats)
    elif model_type == 'nn':
        model = train_nn(xtrn, ytrn, xval, yval, feats)

    # save to disk
    path = configs.model_dir / '{}_fold{}.dat'.format(model_type, fold)
    pickle.dump(model, open(path, "wb"))


def train_cat(xtrain, ytrain, xval, yval, feats):
    xtrain = xtrain[feats].astype(np.float32)
    xval = xval[feats].astype(np.float32)
    cat_cols = ['content_id', 'task_container_id', 'part']

    for col in cat_cols:
        xtrain[col] = xtrain[col].astype(np.int16)
        xval[col] = xval[col].astype(np.int16)

    model = CatBoostClassifier(iterations=4000,
                               learning_rate=0.02,
                               thread_count=4,
                               depth=6,
                               loss_function='Logloss',
                               bootstrap_type='Bernoulli',
                               l2_leaf_reg=40,
                               subsample=0.8,
                               eval_metric='Logloss',
                               metric_period=100,
                               od_type='Iter',
                               od_wait=50,
                               random_seed=0,
                               task_type='GPU')

    model.fit(xtrain, ytrain,
              eval_set=(xval, yval),
              cat_features=cat_cols,
              verbose=True)

    preds = model.predict_proba(xval)[:, 1]
    print('auc:', roc_auc_score(yval, preds))

    return model


def train_epoch(model, loader, optimizer, device, criterion):
    model.train()
    train_loss = 0.0

    for i, (sample, target) in enumerate(tqdm(loader)):
        sample, target = sample.to(device), target.to(device)
        optimizer.zero_grad()

        preds = model(sample)

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
        for i, (sample, target) in enumerate(tqdm(loader)):
            sample, target = sample.to(device), target.to(device)

            preds = model(sample)
            loss = criterion(preds, target)
            val_loss += loss.item() / len(loader)

            val_preds.append(preds.cpu())
            val_targets.append(target.cpu())

        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_auc = roc_auc_score(val_targets, val_preds)

    return val_loss, val_auc


def train_nn(xtrain, ytrain, xval, yval, feats):
    helpers.set_seed(42)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    xtrain, ytrain = xtrain[feats].fillna(-1.0).values.astype(np.float32), ytrain.values
    xval, yval = xval[feats].fillna(-1.0).values.astype(np.float32), yval.values

    train_set = nn_dataset.RiidDataset(xtrain, ytrain)
    val_set = nn_dataset.RiidDataset(xval, yval)

    train_loader = DataLoader(train_set, batch_size=hp.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=hp.val_batch_size, shuffle=False)

    model = nn_model.RiidModel(len(feats)).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, eps=1e-4,
                                                     verbose=True)
    checkpoint_path = configs.model_dir / 'nn'
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
            epoch_path = checkpoint_path / f'epoch{epoch}_{val_auc}.pt'
            torch.save(model.state_dict(), epoch_path)

        # log losses
        writer.add_scalar("Train loss", loss, epoch)
        writer.add_scalar("Val loss", val_loss, epoch)
        writer.add_scalar("Val auc", val_auc, epoch)

        print("Train loss: {:5.5f}".format(loss))
        print("Val loss: {:5.5f}    Val auc: {:4.4f}".format(val_loss, val_auc))

    writer.flush()
    writer.close()

    # return model


def train_lgb(xtrain, ytrain, xval, yval, feats, plot=False):
    if plot:
        xtrain = xtrain[feats]
        xval = xval[feats]
    else:
        xtrain = xtrain[feats].values.astype(np.float32)
        xval = xval[feats].values.astype(np.float32)

    lgb_train = lgb.Dataset(xtrain, ytrain)
    lgb_valid = lgb.Dataset(xval, yval)

    model = lgb.train(
        {'objective': 'binary', 'learning_rate': 0.2, 'feature_fraction': .9, 'num_leaves': 150},
        lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        verbose_eval=100,
        num_boost_round=10000,
        early_stopping_rounds=50,
    )

    preds = model.predict(xval)
    print('auc:', roc_auc_score(yval, preds))

    # show feature importances
    if plot:
        _ = lgb.plot_importance(model)
        plt.show()

        _ = lgb.plot_importance(model, importance_type='gain')
        plt.show()

    return model
