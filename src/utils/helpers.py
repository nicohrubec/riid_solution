import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn

from src.utils import configs


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


def replace_bools(df):
    df.loc[:, 'prior_question_had_explanation'] = df.loc[:, 'prior_question_had_explanation'].map(
        {False: 0, True: 1}
    )
    df['prior_question_had_explanation'] = df['prior_question_had_explanation'].astype(np.float16)

    return df


def filter_train(df, fold):
    val_path = configs.data_dir / 'fold{}_val.csv'.format(fold)
    val = pd.read_csv(val_path)

    val_rows = val.row_id.unique()
    df = df[~df.row_id.isin(val_rows)]

    return df


def load_base_features(fold, mode, tail=False, full=False):
    if mode == 'train':
        if not full:
            fold_path = configs.data_dir / 'fold{}_train.csv'.format(fold)
        else:
            fold_path = configs.all_data_file
    elif mode == 'val':
        fold_path = configs.data_dir / 'fold{}_val.csv'.format(fold)

    df = pd.read_csv(fold_path,
                     dtype={
                         'timestamp': 'int64',
                         'user_id': 'int32',
                         'content_id': 'int16',
                         'content_type_id': 'int8',
                         'answered_correctly': 'int8',
                         'prior_question_elapsed_time': 'float32',
                         'prior_question_had_explanation': 'boolean',
                         'task_container_id': 'int16',
                         'user_answer': 'int8',
                         'part': 'int8'
                     })
    if mode == 'train': df = df[:1000000]
    if mode == 'val': df = df[:1000000]

    del df['content_type_id']
    del df['tags']

    print("Load {}: ".format(mode), df.shape)
    print(df.shape)

    df = df[df.answered_correctly != -1]
    print("Exclude lectures: ", df.shape)

    if tail:
        if mode == 'train':
            df = df.groupby('user_id').tail(1000)
            print("Pick user history tails: ", df.shape)

    if mode == 'train':
        if full:
            df = filter_train(df, fold=fold)
            print("Filter valdiation data from train: ", df.shape)

    target = df['answered_correctly']
    df = replace_bools(df)

    return df, target
