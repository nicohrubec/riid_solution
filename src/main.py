from src.inference import infer_lgb
from src.modeling import train_lgb, train_transformer
from src.preprocessing import merge, split

do_merge = False
do_fold_split = False
do_val_split = False
do_train_lgb = False
do_train_transformer = True
do_inference = False

if do_merge:
    merge.merge_all()
if do_fold_split:
    split.split_to_folds()
if do_val_split:
    split.split_fold(0)
if do_train_lgb:
    train_lgb.train_lgb_fold(0)
if do_train_transformer:
    train_transformer.train_transformer_fold(0)
if do_inference:
    infer_lgb.infer_lgb(0)
