import pickle

import numpy as np
from sklearn.metrics import roc_auc_score

from src.preprocessing import feature_engineering
from src.utils import helpers, configs


def infer_lgb(fold):
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

    oof = infer(xval, yval, feats, fold)
    oof_path = configs.oof_dir / 'oof.npy'
    np.save(oof_path, oof)


def infer(xval, yval, feats, fold):
    model_path = configs.model_dir / 'lgb_7892_fold{}.dat'.format(fold)
    model = pickle.load(open(model_path, "rb"))

    val_preds = model.predict(xval[feats].values.astype(np.float32))
    print('auc:', roc_auc_score(yval, val_preds))

    return val_preds
