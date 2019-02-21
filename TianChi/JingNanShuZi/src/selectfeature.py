'''挑选特征'''

from sklearn.linear_model import RandomizedLasso
from sklearn.ensemble import RandomForestRegressor
import numpy as np


def featureSelect(select_fun, train_data, train_label, threshold_num=0, alpha=0.000015):
    X = train_data  # train data
    Y = train_label  # train label
    feture_names = list(train_data.columns)  # 现有的特征名字
    importance_features_list = []

    if select_fun == 'MeanDecreaseImpurity':
        '''平均不纯度减少 mean decrease impurity'''
        rf = RandomForestRegressor(random_state=2019)
        rf.fit(X, Y)
        feature_score = sorted(zip(feture_names, map(lambda x: round(x, 4), rf.feature_importances_)))
    elif select_fun == 'StabilitySelection':
        '''稳定性选择 StabilitySelection'''
        rlasso = RandomizedLasso(alpha, random_state=2019)  # alpha太大会导致所有特征都会为0，为1最好
        rlasso.fit(X, Y)
        feature_score = sorted(zip(feture_names, map(lambda x: round(x, 4), rlasso.scores_)))
    else:
        importance_features_list = ['MeanDecreaseImpurity', 'StabilitySelection', 'RecursiveFeatureElimination',
                                    'MeanDecreaseAccuracy']
        print("可选挑选特征的方法名:", importance_features_list)
        return importance_features_list

    for item in feature_score:
        if item[1] > threshold_num:
            importance_features_list.append(item[0])
        else:
            continue

    return importance_features_list