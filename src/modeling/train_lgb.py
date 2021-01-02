import pickle

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

from src.preprocessing import feature_engineering
from src.utils import configs
from src.utils import helpers


def train_lgb_fold(fold):
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
    model = train(xtrn, ytrn, xval, yval, feats, plot=False)

    # save to disk
    path = configs.model_dir / 'lgb_fold{}.dat'.format(fold)
    pickle.dump(model, open(path, "wb"))


def train(xtrain, ytrain, xval, yval, feats, plot=False):
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
