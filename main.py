from src.inference import infer_lgb
from src.modeling import train_lgb
from src.preprocessing import merge, split

do_merge = False
do_fold_split = False
do_val_split = False
do_train_lgb = True
do_train_cat = False
do_train_nn = False
do_inference = False
full_data = False

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
