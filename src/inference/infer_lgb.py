import pickle

from sklearn.metrics import roc_auc_score

from src.preprocessing import feature_engineering
from src.utils import helpers, configs


def infer_lgb(fold):
    feats = ['content_id', 'task_container_id', 'prior_question_elapsed_time', 'prior_question_had_explanation', 'part',
             'content_id_target_mean']
    target = 'answered_correctly'
    xtrn, ytrn = helpers.load_base_features(fold, mode='train')
    xval, yval = helpers.load_base_features(fold, mode='val')

    # get features
    xtrn, xval = feature_engineering.get_global_stats(xtrn, xval, target)

    infer(xval, yval, feats, fold)


def infer(xval, yval, feats, fold):
    model_path = configs.model_dir / 'lgb_fold{}.dat'.format(fold)
    model = pickle.load(open(model_path, "rb"))

    val_preds = model.predict(xval[feats])
    print('auc:', roc_auc_score(yval, val_preds))