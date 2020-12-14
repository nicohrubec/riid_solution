from src.preprocessing import merge, split
from src.modeling import train_lgb
from src.inference import infer_lgb

do_merge = False
do_fold_split = False
do_val_split = False
do_training = False
do_inference = True

if do_merge:
    merge.merge_all()
if do_fold_split:
    split.split_to_folds()
if do_val_split:
    split.split_fold(0)
if do_training:
    train_lgb.train_lgb_fold(0)
if do_inference:
    infer_lgb.infer_lgb(0)





