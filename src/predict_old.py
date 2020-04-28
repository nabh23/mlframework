import os
import pandas as pd
from sklearn import preprocessing
from sklearn import ensemble
from sklearn import metrics
import joblib
import numpy as np

from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
MODEL = os.environ.get("MODEL")


"""
if __name__ == "__main__":
    df = pd.read_csv(TEST_DATA)

    for FOLD in range(5):
        encoders = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_label_encoder.pkl"))

        for c in train_df.columns:
            lbl = encoders[c]
            #lbl.fit(train_df[c].values.tolist() + valid_df[c].values.tolist())
            df.loc[:, c] = lbl.transform(df[c].values.tolist())
            

    # data is ready to train
    # clf = ensemble.RandomForestClassifier(n_estimators = 200, n_jobs = -1, verbose=2)
    clf = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}.pkl"))
    cols = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_columns.pkl"))
    

    preds = clf.predict_proba(valid_df)[:, 1]
    #print(metrics.roc_auc_score(yvalid, preds))
"""

def predict():
    df = pd.read_csv(TEST_DATA)
    test_idx = df['id'].values
    predictions = None


    for FOLD in range(5):
        encoders = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_label_encoder.pkl"))
        cols = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_columns.pkl"))
        for c in cols:
            lbl = encoders[c]
            df.loc[:, c] = lbl.transform(df[c].values.tolist())
            

        # data is ready to train
        clf = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}.pkl"))
        cols = joblib.load(os.path.join("models", f"{MODEL}_{FOLD}_columns.pkl"))

        df = df[cols]
        preds = clf.predict_proba(df)[:, 1]

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds

    predictions /= 5

    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns =["id", "target"])
    return sub

if __name__ == "__main__":
    submission = predict()
    submission.to_csv(f"models/{MODEL}.csv", index = False)








