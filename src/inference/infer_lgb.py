import lightgbm as lgb
import pickle
from sklearn.metrics import roc_auc_score

from src.utils import helpers, configs


def infer_lgb(fold):
    base_feats = ['content_id', 'content_type_id', 'task_container_id', 'user_answer',
                  'prior_question_elapsed_time', 'prior_question_had_explanation', 'part']
    xtrn, ytrn = helpers.load_base_features(fold, mode='train')
    xval, yval = helpers.load_base_features(fold, mode='val')

    infer(xval, yval, base_feats, fold)


def infer(xval, yval, feats, fold):
    model_path = configs.model_dir / 'lgb_fold{}.dat'.format(fold)
    model = pickle.load(open(model_path, "rb"))

    val_preds = model.predict(xval[feats])
    print('auc:', roc_auc_score(yval, val_preds))