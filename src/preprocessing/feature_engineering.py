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
    trn[feat_name] = trn[feat_name].astype(np.float32)

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


def get_q_hardness_bins(df):
    df['question_hardness'] = 0
    df.question_hardness.values[df.content_id_target_mean >= 0.9] = -10  # very easy
    df.question_hardness.values[(df.content_id_target_mean >= 0.7) & (df.content_id_target_mean < 0.9)] = -11  # easier
    df.question_hardness.values[(df.content_id_target_mean >= 0.5) & (df.content_id_target_mean < 0.7)] = -12  # harder
    df.question_hardness.values[df.content_id_target_mean < 0.5] = -13  # hard

    return df


def get_global_stats(trn, val, target, save_dicts=False):
    # compute stats merge on train and obtain state dicts for test merge
    trn, content_dict = get_and_merge_feat(trn, target, 'content_id')
    # trn, part_dict = get_and_merge_feat(trn, target, 'part')
    # trn, task_dict = get_and_merge_feat(trn, target, 'task_container_id')

    # merge on validation data
    val['content_id_target_mean'] = merge_feat_val(val['content_id'], content_dict)
    # val['part_target_mean'] = merge_feat_val(val['part'], part_dict)
    # val['task_container_id_target_mean'] = merge_feat_val(val['task_container_id'], task_dict)

    # question hardness bins
    trn = get_q_hardness_bins(trn)
    val = get_q_hardness_bins(val)

    if save_dicts:
        print("Save content dict ...")
        with open(configs.content_dict_path, 'wb') as f:
            pickle.dump(content_dict, f, pickle.HIGHEST_PROTOCOL)

    return trn, val


def update_dicts(row, count_dict, correct_dict, time_dict, last_n_dict):
    # get dictionary keys for each feature
    user = int(row[0])
    question = int(row[1])
    timestamp = int(row[2])
    part = int(-row[3])
    task_id = int(row[4])
    hardness = int(row[5])
    correct = int(row[6])

    if user in count_dict:  # known user
        # overall user features
        count_dict[user]['sum'] += 1
        correct_dict[user]['sum'] += correct
        time_dict[user]['last'] = timestamp
        last_n_dict[user]['last_task2'] = last_n_dict[user]['last_task']
        last_n_dict[user]['last_task'] = task_id

        # update rolling answered correct for user
        last_n_dict[user]['last_n'].append(correct)
        correction = last_n_dict[user]['last_n'].pop(0)
        last_n_dict[user]['sum'] -= correction
        last_n_dict[user]['sum'] += correct

        # update last n timestamps for user
        last_n_dict[user]['last_n_time'].append(timestamp)
        last_n_dict[user]['time_sum'] = last_n_dict[user]['last_n_time'].pop(0)
        last_n_dict[user]['time_sum2'] = last_n_dict[user]['last_n_time'][9]
        last_n_dict[user]['time_sum3'] = last_n_dict[user]['last_n_time'][14]

        # update question difficulty specific features
        if hardness in count_dict[user]:
            count_dict[user][hardness] += 1
            correct_dict[user][hardness] += correct
        else:
            count_dict[user][hardness] = 1
            correct_dict[user][hardness] = correct

        # update part specific features
        if part in count_dict[user]:
            count_dict[user][part] += 1
            correct_dict[user][part] += correct
        else:
            count_dict[user][part] = 1
            correct_dict[user][part] = correct

        # update question specific features
        if question in count_dict[user]:  # known question for this user
            count_dict[user][question] += 1
            correct_dict[user][question] += correct
            time_dict[user][question] = timestamp
        else:  # unknown question for this user
            count_dict[user][question] = 1
            correct_dict[user][question] = correct
            time_dict[user][question] = timestamp

    else:  # unknown user create new entry
        count_dict[user] = {'sum': 1, question: 1, part: 1, hardness: 1}
        correct_dict[user] = {'sum': correct, question: correct, part: correct, hardness: correct}
        time_dict[user] = {'last': timestamp, question: timestamp}
        last_n_dict[user] = {'sum': correct, 'time_sum': 0, 'time_sum2': 0, 'time_sum3': 0,
                             'last_n': [0, 0, 0, 0, correct], 'last_task': task_id, 'last_task2': np.nan,
                             'last_n_time': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, timestamp]}

    return count_dict, correct_dict, time_dict, last_n_dict


def get_row_values(row, count_dict, correct_dict, time_dict, last_n_dict):
    feats = np.full(23, fill_value=np.nan, dtype=np.float32)  # row storage

    # get dictionary keys
    user = int(row[0])
    question = int(row[1])
    timestamp = int(row[2])
    part = int(-row[3])
    task_id = int(row[4])
    hardness = int(row[5])

    if user in count_dict:  # known user
        # get overall user specific features
        feats[0] = count_dict[user]['sum']
        feats[1] = correct_dict[user]['sum']
        feats[4] = timestamp - time_dict[user]['last']
        feats[6] = last_n_dict[user]['sum']
        feats[7] = timestamp - last_n_dict[user]['time_sum']
        feats[8] = timestamp - last_n_dict[user]['time_sum2']
        feats[9] = timestamp - last_n_dict[user]['time_sum3']
        feats[19] = task_id - last_n_dict[user]['last_task']
        feats[20] = task_id - last_n_dict[user]['last_task2']

        # get question difficulty specific features
        if hardness in count_dict[user]:
            feats[21] = count_dict[user][hardness]
            feats[22] = correct_dict[user][hardness]

        # get part specific features
        if part in count_dict[user]:
            feats[10] = count_dict[user][part]
            feats[11] = correct_dict[user][part]

        # add part features for all other parts as well
        for i, p in enumerate([-1, -2, -3, -4, -5, -6, -7]):
            if p in count_dict[user]:
                feats[12 + i] = correct_dict[user][p] / count_dict[user][p]

        # add question specific features
        if question in count_dict[user]:  # known question for this user
            feats[2] = count_dict[user][question]
            feats[3] = correct_dict[user][question]
            feats[5] = timestamp - time_dict[user][question]

    return feats


def calc_feats_from_stats(df, user_feats):
    # compute correctness means
    user_feats[:, 1] = user_feats[:, 1] / user_feats[:, 0]
    user_feats[:, 3] = user_feats[:, 3] / user_feats[:, 2]
    user_feats[:, 11] = user_feats[:, 11] / user_feats[:, 10]
    user_feats[:, 22] = user_feats[:, 22] / user_feats[:, 21]

    # assign computed features to new columns in the df
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
    df['user_last_n_time'] = user_feats[:, 7].astype(np.float32)
    df['user_last_n_time2'] = user_feats[:, 8].astype(np.float32)
    df['user_last_n_time3'] = user_feats[:, 9].astype(np.float32)
    df['user_part_count'] = user_feats[:, 10].astype(np.float32)
    df['user_part_correct_mean'] = user_feats[:, 11].astype(np.float32)
    df['user_part1_mean'] = user_feats[:, 12].astype(np.float32)
    df['user_part2_mean'] = user_feats[:, 13].astype(np.float32)
    df['user_part3_mean'] = user_feats[:, 14].astype(np.float32)
    df['user_part4_mean'] = user_feats[:, 15].astype(np.float32)
    df['user_part5_mean'] = user_feats[:, 16].astype(np.float32)
    df['user_part6_mean'] = user_feats[:, 17].astype(np.float32)
    df['user_part7_mean'] = user_feats[:, 18].astype(np.float32)
    df['task_container_eq1'] = user_feats[:, 19].astype(np.float32)
    df['task_container_eq2'] = user_feats[:, 20].astype(np.float32)
    df['user_hardness_count'] = user_feats[:, 21].astype(np.float32)
    df['user_hardness_mean'] = user_feats[:, 22].astype(np.float32)
    df['user_hardness_inter'] = df['user_hardness_mean'] - df['content_id_target_mean']

    return df


def calc_dicts_and_add(df, count_dict=None, correct_dict=None, time_dict=None, last_n_dict=None):
    # init empty dicts if nothing is provided else use precomputed dicts
    # count_dict: user specific counts for questions, parts, question difficulty etc.
    # correct_dict: user specific counts of correct answers for questions, parts, question difficulty etc.
    # time dict: last time we saw the user, last time user answered this question etc.
    # last n dict: last n time user correct, last n time user question correct, last n time seen user etc.
    if not count_dict:
        count_dict = {}
    if not correct_dict:
        correct_dict = {}
    if not time_dict:
        time_dict = {}
    if not last_n_dict:
        last_n_dict = {}

    # init numpy storage for all features and create row iterator
    user_feats = np.full((len(df), 23), fill_value=np.nan, dtype=np.float32)
    prev_row = None
    feat_iterator = df[['user_id', 'content_id', 'timestamp', 'part', 'task_container_id', 'question_hardness',
                        'answered_correctly']].values
    del df['timestamp']
    del df['user_id']
    del df['row_id']

    for row_id, curr_row in enumerate(tqdm(feat_iterator)):
        if prev_row is not None:
            # increment user information incrementally with newly gained information (previous row)
            count_dict, correct_dict, time_dict, last_n_dict = update_dicts(prev_row, count_dict, correct_dict,
                                                                            time_dict, last_n_dict)

        # obtain feature values for current row
        user_row_values = get_row_values(curr_row, count_dict, correct_dict, time_dict, last_n_dict)
        user_feats[row_id] = user_row_values

        prev_row = curr_row

    # calculate and add features from preprocessed state dicts to data
    del feat_iterator
    df = calc_feats_from_stats(df, user_feats)

    return df, count_dict, correct_dict, time_dict, last_n_dict


def get_user_feats(trn, val, save_dicts=False):
    # build up the feature dicts row wise on train and compute train features
    trn, count_dict, correct_dict, time_dict, last_n_dict = calc_dicts_and_add(trn)
    # add to precomputed train feature dicts and compute inference features
    val, count_dict, correct_dict, time_dict, last_n_dict = calc_dicts_and_add(val, count_dict, correct_dict,
                                                                               time_dict, last_n_dict)

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
        print("Save last n dict ...")
        with open(configs.last_n_dict_path, 'wb') as f:
            pickle.dump(last_n_dict, f, pickle.HIGHEST_PROTOCOL)

    del count_dict, correct_dict, time_dict, last_n_dict
    gc.collect()

    return trn, val


def get_answer_feats(trn, val):
    answer_counts = trn.groupby('content_id')['user_answer'].value_counts(normalize=True)

    answer_counts_unstack = answer_counts.unstack().reset_index(drop=True).astype(np.float32)
    answer_counts_unstack.columns = ['answer1', 'answer2', 'answer3', 'answer4']
    answer_counts_unstack = answer_counts_unstack.rename_axis('content_id').reset_index()
    answers = answer_counts_unstack.values[:, -4:].astype(np.float32)
    answers.sort(axis=1)
    answer_counts_unstack[['answer1', 'answer2', 'answer3', 'answer4']] = answers
    answer_counts_unstack = answer_counts_unstack.astype(np.float32)

    trn = pd.merge(trn, answer_counts_unstack, how='left', on='content_id')
    val = pd.merge(val, answer_counts_unstack, how='left', on='content_id')

    return trn, val
