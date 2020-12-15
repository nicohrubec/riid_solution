import pickle

import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from src.preprocessing import feature_engineering
from src.utils import configs
from src.utils import helpers


def train_lgb_fold(fold):
    # get base features
    feats = ['content_id', 'task_container_id', 'prior_question_elapsed_time', 'prior_question_had_explanation', 'part',
             'content_id_target_mean']
    target = 'answered_correctly'
    xtrn, ytrn = helpers.load_base_features(fold, mode='train')
    xval, yval = helpers.load_base_features(fold, mode='val')

    # get features
    xtrn, xval = feature_engineering.get_global_stats(xtrn, xval, target)
    # get row wise user stats etc --> up to point of row

    # train model on single fold
    model = train(xtrn, ytrn, xval, yval, feats)

    # save to disk
    path = configs.model_dir / 'lgb_fold{}.dat'.format(fold)
    pickle.dump(model, open(path, "wb"))


def train(xtrain, ytrain, xval, yval, feats, plot=False):
    lgb_train = lgb.Dataset(xtrain[feats], ytrain)
    lgb_valid = lgb.Dataset(xval[feats], yval)

    model = lgb.train(
        {'objective': 'binary'},
        lgb_train,
        valid_sets=[lgb_train, lgb_valid],
        verbose_eval=100,
        num_boost_round=10000,
        early_stopping_rounds=10,
    )

    preds = model.predict(xval[feats])
    print('auc:', roc_auc_score(yval, preds))

    # show feature importances
    if plot:
        _ = lgb.plot_importance(model)
        plt.show()

    return model
