# -*-coding: utf-8-*-
# @Author  : Charlesxu
# @Email   : charlesxu.ai@gmail.com

import pandas as pd
import numpy as np


def getHit(df):
    df = df.sort_values('pred_y', ascending=False).reset_index()
    if df[df.true_y == 1].index.tolist()[0] < _K:
        return 1
    else:
        return 0


def getNDCG(df):
    df = df.sort_values('pred_y', ascending=False).reset_index()
    i = df[df.true_y == 1].index.tolist()[0]
    if i < _K:
        return np.log(2) / np.log(i+2)
    else:
        return 0.


def evaluate_model(model, test, K):
    global _K
    _K = K
    test_X, test_y = test
    pred_y = model.predict(test_X)
    test_df = pd.DataFrame(test_y, columns=['user_id', 'true_y'])
    test_df['pred_y'] = pred_y
    tg = test_df.groupby('user_id')
    hit_rate = tg.apply(getHit).mean()
    ndcg = tg.apply(getNDCG).mean()
    return hit_rate, ndcg