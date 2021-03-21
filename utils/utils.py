# -*-coding: utf-8-*-
# @Author  : Charlesxu
# @Email   : charlesxu.ai@gmail.com

import pandas as pd
import random
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences


def sparseFeature(feat, feat_num, embed_dim=4):
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def create_implicit_ml_1m_dataset(file, trans_score=2, embed_dim=8, maxlen=40):
    data_df = pd.read_csv(file, sep="::", engine='python',
                          names=['user_id', 'item_id', 'label', 'Timestamp'])
    data_df = data_df[data_df.label >= trans_score]

    data_df = data_df.sort_values(by=['user_id', 'Timestamp'])

    train_data, val_data, test_data = [], [], []

    item_id_max = data_df['item_id'].max()
    for user_id, df in tqdm(data_df[['user_id', 'item_id']].groupby('user_id')):
        pos_list = df['item_id'].tolist()

        def gen_neg():
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(1, item_id_max)
                return neg

        neg_list = [gen_neg() for i in range(len(pos_list) + 100)]
        for i in range(1, len(pos_list)):
            hist_i = pos_list[:i]
            if i == len(pos_list) - 1:
                test_data.append([hist_i, pos_list[i], [user_id, 1]])
                for neg in neg_list[i:]:
                    test_data.append([hist_i, neg, [user_id, 0]])
            elif i == len(pos_list) - 2:
                val_data.append([hist_i, pos_list[i], 1])
                val_data.append([hist_i, neg_list[i], 0])
            else:
                train_data.append([hist_i, pos_list[i], 1])
                train_data.append([hist_i, neg_list[i], 0])
    user_num, item_num = data_df['user_id'].max() + 1, data_df['item_id'].max() + 1
    item_feat_col = sparseFeature('item_id', item_num, embed_dim)

    random.shuffle(train_data)
    random.shuffle(val_data)

    train = pd.DataFrame(train_data, columns=['hist', 'target_item', 'label'])
    val = pd.DataFrame(val_data, columns=['hist', 'target_item', 'label'])
    test = pd.DataFrame(test_data, columns=['hist', 'target_item', 'label'])

    train_X = [pad_sequences(train['hist'], maxlen=maxlen), train['target_item'].values]
    train_y = train['label'].values
    val_X = [pad_sequences(val['hist'], maxlen=maxlen), val['target_item'].values]
    val_y = val['label'].values
    test_X = [pad_sequences(test['hist'], maxlen=maxlen), test['target_item'].values]
    test_y = test['label'].values.tolist()
    return item_feat_col, (train_X, train_y), (val_X, val_y), (test_X, test_y)


