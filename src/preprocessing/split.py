import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import GroupKFold
import random
import gc

from src.utils import configs
from src.utils.hyperparameters import nfolds, usage_fold, holdout_fold, unseen_fold


# splits the master df into 5 folds by user
def split_to_folds():
    data = pd.read_csv(configs.all_data_file,
                       dtype={
                           'timestamp': 'int64',
                           'user_id': 'int32',
                           'content_id': 'int16',
                           'content_type_id': 'int8',
                           'answered_correctly': 'int8',
                           'prior_question_elapsed_time': 'float32',
                           'prior_question_had_explanation': 'boolean',
                           'task_container_id': 'int16',
                           'user_answer': 'int8'
                       })

    gkf = GroupKFold(n_splits=nfolds)
    for fold, (tr, te) in enumerate(gkf.split(data, data, groups=data.user_id)):
        print("Saving fold {}".format(fold))
        fold_data = data.iloc[te]
        print("Shape: {}".format(fold_data.shape))

        fold_path = configs.data_dir / ('fold'+str(fold)+'.csv')
        fold_data.to_csv(fold_path, index=False)


def rand_time(max_time_stamp_all_users, max_time_stamp_user):
    interval = max_time_stamp_all_users - max_time_stamp_user
    rand_time_stamp = random.randint(0, interval)
    return rand_time_stamp


# splits a given fold into training and validation data based on specified hyperparameters -->
# usage_fold: proportion the data we want to use
# holdout_fold: proportion of remaining data we want to use as validation data
# unseen_fold: proportion of users in validation data that is unseen during training time
# training and validation data are written to disk in seperate dfs for later usage
def split_fold(fold):
    print("Splitting fold {}".format(fold))
    fold_path = configs.data_dir / ('fold'+str(fold)+'.csv')
    fold_df = pd.read_csv(fold_path,
                          dtype={
                              'timestamp': 'int64',
                              'user_id': 'int32',
                              'content_id': 'int16',
                              'content_type_id': 'int8',
                              'answered_correctly': 'int8',
                              'prior_question_elapsed_time': 'float32',
                              'prior_question_had_explanation': 'boolean',
                              'task_container_id': 'int16',
                              'user_answer': 'int8'
                          })

    # select subset of users to decrease data size
    train_users = fold_df.user_id.unique()
    num_users = int(len(train_users) * usage_fold)
    train_users = train_users[:num_users]
    fold_df = fold_df[fold_df.user_id.isin(train_users)]

    max_timestamp_u = fold_df[['user_id', 'timestamp']].groupby(['user_id']).agg(['max']).reset_index()
    max_timestamp_u.columns = ['user_id', 'max_time_stamp']
    MAX_TIME_STAMP = max_timestamp_u.max_time_stamp.max()

    max_timestamp_u['rand_time_stamp'] = max_timestamp_u.max_time_stamp.apply(
        lambda user_time: rand_time(MAX_TIME_STAMP, user_time))
    fold_df = fold_df.merge(max_timestamp_u, on='user_id', how='left')
    fold_df['virtual_time_stamp'] = fold_df.timestamp + fold_df['rand_time_stamp']

    del fold_df['max_time_stamp']
    del fold_df['rand_time_stamp']
    del max_timestamp_u
    gc.collect()

    fold_df = fold_df.sort_values(['virtual_time_stamp', 'row_id']).reset_index(drop=True)
    del fold_df['virtual_time_stamp']
    train_size = int(len(fold_df) * (1-holdout_fold))

    train = fold_df[:train_size]
    val = fold_df[train_size:]

    print("Shape of training data: ", train.shape)
    print("Shape of validation data: ", val.shape)

    # save
    train_path = configs.data_dir / ('fold' + str(fold) + '_train.csv')
    train.to_csv(train_path, index=False)
    val_path = configs.data_dir / ('fold' + str(fold) + '_val.csv')
    val.to_csv(val_path, index=False)


def split_all_folds():
    for fold in range(nfolds):
        split_fold(fold)
