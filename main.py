from src.inference import infer_lgb
from src.modeling import train_lgb
from src.preprocessing import merge, split

do_merge = False  # merge lecture and question file to base train data
do_fold_split = False  # split 100 mio rows to smaller chunks of 20 mio each for prototyping
do_val_split = False  # split a chunk into train and validation data
do_train_lgb = True  # train single fold lgb
do_train_cat = False  # train single fold catboost
do_train_nn = False  # train single fold linear nn
do_inference = False  # run only inference on lgb without training
full_data = False  # if true uses 100 mio row, false use 18 mio rows in current setup

if do_merge:
    merge.merge_all()
if do_fold_split:
    split.split_to_folds()
if do_val_split:
    split.split_fold(0)
if do_train_lgb:
    train_lgb.train_fold(0, model_type='lgb', full_data=full_data)
if do_train_cat:
    train_lgb.train_fold(0, model_type='cat', full_data=full_data)
if do_train_nn:
    train_lgb.train_fold(0, model_type='nn', full_data=full_data)
if do_inference:
    infer_lgb.infer_lgb(0)
