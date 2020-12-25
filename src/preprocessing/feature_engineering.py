import gc
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils import configs


def get_and_merge_feat(trn, target, feat):
    # compute target mean and merge on train
    feat_name = '{}_target_mean'.format(feat)
    mean = trn[[feat, target]].groupby([feat]).agg(['mean'])
    mean.columns = [feat_name]
    trn = pd.merge(trn, mean, on=feat, how='left')

    # transform df to dict for test merge
    feat_dict = mean.astype('float32').to_dict()[feat_name]

    return trn, feat_dict


def merge_feat_val(key_feat, feat_dict):
    add_feat = np.zeros((len(key_feat)))
    key_feat = key_feat.values

    for row_id, row in enumerate(key_feat):
        key = key_feat[row_id]

        if key in feat_dict:
            add_feat[row_id] = feat_dict[key]
        else:
            add_feat[row_id] = -1

    return add_feat


def get_global_stats(trn, val, target, save_dicts=False):
    # compute stats merge on train and obtain state dicts for test merge
    trn, content_dict = get_and_merge_feat(trn, target, 'content_id')
    # trn, part_dict = get_and_merge_feat(trn, target, 'part')
    # trn, task_dict = get_and_merge_feat(trn, target, 'task_container_id')

    # merge on validation data
    val['content_id_target_mean'] = merge_feat_val(val['content_id'], content_dict)
    # val['part_target_mean'] = merge_feat_val(val['part'], part_dict)
    # val['task_container_id_target_mean'] = merge_feat_val(val['task_container_id'], task_dict)

    if save_dicts:
        print("Save content dict ...")
        with open(configs.content_dict_path, 'wb') as f:
            pickle.dump(content_dict, f, pickle.HIGHEST_PROTOCOL)

    return trn, val


def update_dicts(row, count_dict, correct_dict, time_dict, last_n_dict):
    user = int(row[0])
    question = int(row[1])
    timestamp = int(row[2])
    correct = int(row[3])

    if user in count_dict:  # known user
        count_dict[user]['sum'] += 1
        correct_dict[user]['sum'] += correct
        time_dict[user]['last'] = timestamp

        last_n_dict[user]['last_n'].append(correct)
        correction = last_n_dict[user]['last_n'].pop(0)
        last_n_dict[user]['sum'] -= correction
        last_n_dict[user]['sum'] += correct

        if question in count_dict[user]:  # known question for this user
            count_dict[user][question] += 1
            correct_dict[user][question] += correct
            time_dict[user][question] = timestamp
        else:  # unknown question for this user
            count_dict[user][question] = 1
            correct_dict[user][question] = correct
            time_dict[user][question] = timestamp
    else:  # unknown user
        count_dict[user] = {'sum': 1, question: 1}
        correct_dict[user] = {'sum': correct, question: correct}
        time_dict[user] = {'last': timestamp, question: timestamp}
        last_n_dict[user] = {'sum': correct, 'last_n': [0, 0, 0, 0, correct]}

    return count_dict, correct_dict, time_dict, last_n_dict


def get_row_values(row, count_dict, correct_dict, time_dict, last_n_dict):
    feats = np.zeros(7)
    user = int(row[0])
    question = int(row[1])
    timestamp = int(row[2])

    if user in count_dict:  # known user
        feats[0] = count_dict[user]['sum']
        feats[1] = correct_dict[user]['sum']
        feats[4] = timestamp - time_dict[user]['last']
        feats[6] = last_n_dict[user]['sum']

        if question in count_dict[user]:  # known question for this user
            feats[2] = count_dict[user][question]
            feats[3] = correct_dict[user][question]
            feats[5] = timestamp - time_dict[user][question]
        else:  # unknown question for this user
            feats[2] = 0
            feats[3] = -1
            feats[5] = -1
    else:  # unknown user
        feats[0] = -1
        feats[1] = -1
        feats[2] = -1
        feats[3] = -1
        feats[4] = -1
        feats[5] = -1
        feats[6] = -1

    return feats


def calc_feats_from_stats(df, user_feats):
    # assign computed features to new columns in the df
    user_feats[:, 1] = user_feats[:, 1] / user_feats[:, 0]
    user_feats[:, 3] = user_feats[:, 3] / user_feats[:, 2]

    df['user_count'] = user_feats[:, 0].astype(np.float32)
    df['user_correct_mean'] = user_feats[:, 1].astype(np.float32)
    df['user_question_count'] = user_feats[:, 2].astype(np.float32)
    user_feats[:, 3][user_feats[:, 3] == -np.inf] = 0
    df['user_question_correct_mean'] = user_feats[:, 3].astype(np.float32)
    df['last_time_user'] = user_feats[:, 4].astype(np.float32)
    df['last_time_question_user'] = user_feats[:, 5].astype(np.float32)
    df['last_time_user_inter'] = df['last_time_user'].astype(np.float32) - df['last_time_question_user'].astype(
        np.float32)
    df['user_last_n_correct'] = user_feats[:, 6].astype(np.float32)

    return df


def calc_dicts_and_add(df, count_dict=None, correct_dict=None, time_dict=None, last_n_dict=None):
    # init empty dicts if nothing is provided
    if not count_dict:
        count_dict = {}
    if not correct_dict:
        correct_dict = {}
    if not time_dict:
        time_dict = {}
    if not last_n_dict:
        last_n_dict = {}

    # init numpy storage for all features
    # [user count, user correct count, user question count, user question correct count]
    user_feats = np.zeros((len(df), 7))
    prev_row = None

    # count_dict = {user: {question_counts, user_overall_count}
    # correct_dict = {user: {question_correct counts, user_overall correct_count}}

    for row_id, curr_row in enumerate(tqdm(df[['user_id', 'content_id', 'timestamp', 'answered_correctly']].values)):
        if prev_row is not None:
            # increment user information
            count_dict, correct_dict, time_dict, last_n_dict = update_dicts(prev_row, count_dict, correct_dict,
                                                                            time_dict, last_n_dict)

        # obtain feature values for this row
        user_row_values = get_row_values(curr_row, count_dict, correct_dict, time_dict, last_n_dict)
        user_feats[row_id] = user_row_values

        prev_row = curr_row

    del df['user_answer']
    del df['timestamp']
    del df['user_id']
    # calculate and add features from preprocessed stat dicts
    df = calc_feats_from_stats(df, user_feats)

    return df, count_dict, correct_dict, time_dict, last_n_dict


def get_user_feats(trn, val, save_dicts=False):
    trn, count_dict, correct_dict, time_dict, last_n_dict = calc_dicts_and_add(trn)
    val, count_dict, correct_dict, time_dict, last_n_dict = calc_dicts_and_add(val, count_dict, correct_dict, time_dict,
                                                                               last_n_dict)

    if save_dicts:
        print("Save count dict ...")
        with open(configs.count_dict_path, 'wb') as f:
            pickle.dump(count_dict, f, pickle.HIGHEST_PROTOCOL)
        print("Save correct dict ...")
        with open(configs.correct_dict_path, 'wb') as f:
            pickle.dump(correct_dict, f, pickle.HIGHEST_PROTOCOL)
        print("Save time dict ...")
        with open(configs.time_dict_path, 'wb') as f:
            pickle.dump(time_dict, f, pickle.HIGHEST_PROTOCOL)

    del count_dict, correct_dict, time_dict, last_n_dict
    gc.collect()

    return trn, val
