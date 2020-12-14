import pandas as pd

from src.utils import configs


def replace_bools(df):
    df.loc[:, 'prior_question_had_explanation'] = df.loc[:, 'prior_question_had_explanation'].map(
        {False: 0, True: 1}
    )

    return df


def load_base_features(fold, mode):
    if mode == 'train':
        fold_path = configs.data_dir / 'fold{}_train.csv'.format(fold)
    elif mode == 'val':
        fold_path = configs.data_dir / 'fold{}_val.csv'.format(fold)

    df = pd.read_csv(fold_path)
    df = df[df.answered_correctly != -1]
    target = df['answered_correctly']
    df = replace_bools(df)

    return df, target
