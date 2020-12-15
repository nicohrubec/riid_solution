import numpy as np
import pandas as pd


def get_and_merge_feat(trn, target, feat):
    # compute target mean and merge on train
    feat_name = '{}_target_mean'.format(feat)
    mean = trn[[feat, target]].groupby([feat]).agg(['mean'])
    mean.columns = [feat_name]
    trn = pd.merge(trn, mean, on=feat, how='left')

    # transform df to dict for test merge
    feat_dict = mean.to_dict()[feat_name]

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


def get_global_stats(trn, val, target):
    # compute stats merge on train and obtain state dicts for test merge
    trn, content_dict = get_and_merge_feat(trn, target, 'content_id')
    # trn, part_dict = get_and_merge_feat(trn, target, 'part')
    # trn, task_dict = get_and_merge_feat(trn, target, 'task_container_id')

    # merge on validation data
    val['content_id_target_mean'] = merge_feat_val(val['content_id'], content_dict)
    # val['part_target_mean'] = merge_feat_val(val['part'], part_dict)
    # val['task_container_id_target_mean'] = merge_feat_val(val['task_container_id'], task_dict)

    return trn, val
