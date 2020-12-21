import pandas as pd

from src.utils import configs
from src.utils import hyperparameters as hp


def replace_bools(df):
    df.loc[:, 'prior_question_had_explanation'] = df.loc[:, 'prior_question_had_explanation'].map(
        {False: 0, True: 1}
    )

    return df


def add_remaining_data(df, fold):
    for i in range(hp.nfolds):
        print("Add data for fold {}".format(i))
        print(df.shape)
        if i == fold:
            continue
        fold_path = configs.data_dir / 'fold{}.csv'.format(fold)
        fold_df = pd.read_csv(fold_path)
        fold_df = fold_df.groupby('user_id').tail(1000)

        df = df.append(fold_df)

    del fold_df
    return df


def load_base_features(fold, mode, tail=True, full=True):
    if mode == 'train':
        fold_path = configs.data_dir / 'fold{}_train.csv'.format(fold)
    elif mode == 'val':
        fold_path = configs.data_dir / 'fold{}_val.csv'.format(fold)

    df = pd.read_csv(fold_path)

    if mode == 'train':
        if full:
            df = add_remaining_data(df, fold=fold)

    df = df[df.answered_correctly != -1]
    print(df.shape)

    # only train on tails of user histories
    if tail:
        if mode == 'train':
            if not full:
                df = df.groupby('user_id').tail(1000)
                print(df.shape)

    target = df['answered_correctly']
    df = replace_bools(df)

    return df, target
