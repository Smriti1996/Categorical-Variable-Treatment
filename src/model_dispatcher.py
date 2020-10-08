from sklearn import linear_model
import xgboost as xgb

models = {
    'log_reg': linear_model.LogisticRegression(),
    'xgboost_classifier': xgb.XGBClassifier(n_jobs=-1)
}