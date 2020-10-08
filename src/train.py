import pandas as pd

import joblib
import os
import argparse

import config
import model_dispatcher

from sklearn import metrics
from sklearn import preprocessing

def run(fold, model):

    # load the full training data with folds
    df = pd.read_csv(config.TRAINING_FILE)

    # all columns are features except id, target and kfold columns
    features = [
        f for f in df.columns if f not in ("id", "target", "kfold")
    ]

    # fill all NaN values with NONE 
    # all columns are converted to string
    # it does not matter because all are categories
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)


    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # initialize OneHotEncoder from sciket-learn
    ohe = preprocessing.OneHotEncoder()

    # fit ohe on training + validation features
    full_data = pd.concat(
        [df_train[features] , df_valid[features]],
        axis=0
    )


    ohe.fit(full_data[features])

    # transform training data
    x_train = ohe.transform(df_train[features])

    # transform validation data
    x_valid = ohe.transform(df_valid[features])

    # initialize logistic regression model
    log_reg = model_dispatcher.models[model]

    # fit model on traning data (ohe)
    log_reg. fit(x_train, df_train.target.values)

    # predict on validation data 
    # we need the probability values as we are calculating AUC
    # we will use the probability of 1s
    valid_preds = log_reg.predict_proba(x_valid)[:, 1]

    # get roc and auc score
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    # print auc
    print(f"Fold = {fold}, AUC = {auc}")

    # save the model
    joblib.dump(
        log_reg,
        os.path.join(config.MODEL_OUTPUT, f"dt_{fold}.bin")
    )

if __name__ == "__main__":
    # initialize ArgumentParser class of argparse
    parser = argparse.ArgumentParser()

    # add different arguments you need and their type
    parser.add_argument(
        "--fold",
        type=int
    )

    parser.add_argument(
        "--model",
        type=str
    )

    # read the arguments from command line
    args = parser.parse_args()

    # run the folds specified by command line arguments
    run(
        fold=args.fold,
        model=args.model
    )