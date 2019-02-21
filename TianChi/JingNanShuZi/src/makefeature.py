
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder

from selectfeature import featureSelect


class MakeFeature:

    def __init__(self,data,train_length,target):
        # 会进行类别转换的列
        self.categorical_columns = [f for f in data.columns if f not in ['样本id']]
        # 保留数字特征的列
        self.numerical_columns = [f for f in data.columns if f not in self.categorical_columns]
        # 保留新增特征
        self.new_list = []
        self.mean_columns = []
        # 初始化变量
        self.data = data
        self.train_length = train_length
        self.target = target


    def makeFeature(self):

        process_col = ['B9', 'B10', 'B11']
        self.data['count'] = 0
        for col in process_col:
            self.data['count'] = self.data['count'] + self.data[col].apply(self.countTimes)
        self.categorical_columns.append('count')

        make_feature1_col_list = [["A17","A15"],["A15","A12"],["A12","A10"],["A10","A6"]]
        for i in range(len(make_feature1_col_list)):
            self.make_feature1(make_feature1_col_list[i][0],make_feature1_col_list[i][1])

        self.make_feature2(self.data)
        self.make_feature3(self.data)
        # 将新的特征加入数字特征中
        self.numerical_columns = self.numerical_columns + self.new_list
        # 空值处理
        self.data = self.data.replace([np.inf, -np.inf], np.nan)
        self.data = self.data.fillna(-1)
        # data[data.isin([np.nan, np.inf, -np.inf]).any(1)].shape
        # 类别编码
        self.labelEncoder(self.data)
        self.train_data = self.data[:self.train_length]
        self.test_data = self.data[self.train_length:]
        # print("2", self.train_data.iloc[3])
        self.train_data,self.test_data = self.labelBox(self.train_data,self.test_data)
        # print(self.train_data[self.train_data.isin([np.nan, np.inf, -np.inf]).any(1)].shape)
        importance_features = self._selectFeature()
        del_col_list = ['A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']
        self.test_data = self.test_data.fillna(-1)
        X_train,X_test = self.makeOneHot(self.data, importance_features, del_col_list)

        return X_train,X_test

    def makeOneHot(self,data,importance_features,del_col_list):

        mean_columns_important = list(set(self.mean_columns) & set(importance_features))
        numerical_columns_important = list(set(self.numerical_columns) & set(importance_features))
        categorical_columns_important = list(set(self.categorical_columns) & set(importance_features))
        categorical_columns_important = list(set(categorical_columns_important) - set(del_col_list))

        no_hot_col_list = list(np.sort(mean_columns_important + numerical_columns_important))
        X_train = self.train_data[no_hot_col_list].values
        X_test = self.test_data [no_hot_col_list].values
        # one hot
        enc = OneHotEncoder()
        noehot_col_list = list(np.sort(categorical_columns_important))
        for f in noehot_col_list:
            enc.fit(data[f].values.reshape(-1, 1))
            X_train = sparse.hstack((X_train, enc.transform(self.train_data[f].values.reshape(-1, 1))), 'csr')
            X_test = sparse.hstack((X_test, enc.transform(self.test_data [f].values.reshape(-1, 1))), 'csr')
        print(X_train.shape)
        print(X_test.shape)
        return X_train,X_test

    def _selectFeature(self):
        importance_features = featureSelect('StabilitySelection', self.train_data, self.target, 0, 0.000029)
        print("importance_features:",len(importance_features))
        return importance_features

    def labelEncoder(self,data):
        for f in self.categorical_columns:
            data[f] = data[f].map(dict(zip(data[f].unique(), range(0, data[f].nunique()))))

    def labelBox(self,train,test):
        train['target'] = self.target
        train['intTarget'] = pd.cut(train['target'], 5, labels=False)
        train = pd.get_dummies(train, columns=['intTarget'])
        li = list(train.columns)[-5:]
        count = 0
        for f1 in self.categorical_columns:
            cate_rate = train[f1].value_counts(normalize=True, dropna=False).values[0]
            #     if rate < 0.90:
            for f2 in li:
                col_name = 'B14_to_' + f1 + "_" + f2 + '_mean'
                self.mean_columns.append(col_name)
                order_label = train.groupby([f1])[f2].mean()
                train[col_name] = train['B14'].map(order_label)
                # train[col_name] = train[f1].map(order_label)
                miss_rate = train[col_name].isnull().sum() * 100 / train[col_name].shape[0]
                if miss_rate > 0:
                    count = count+1
                    train = train.drop([col_name], axis=1)
                    self.mean_columns.remove(col_name)
                else:
                    test[col_name] = test['B14'].map(order_label)
            #  test[col_name] = test[f1].map(order_label)
        train.drop(li + ['target'], axis=1, inplace=True)
        return train,test

    def countTimes(self,time):
        count = 0
        if not pd.isna(time):
            count += 1
        return count

    def make_feature1(self,feature_name1, feature_name2):
        feature_name = feature_name1 + "_diff"
        self.data[feature_name] = self.data[feature_name1] - self.data[feature_name2]
        gp = self.data.groupby(feature_name)['样本id'].count()
        self.data[feature_name + '_' + feature_name2 + '_count'] = self.data[feature_name].map(gp) / self.data['B14']
        self.new_list.append(feature_name + "_" + feature_name2 + '_count')

    def make_feature2(self,data):
        data[['A1', 'A3', 'A4', 'A19', 'B1', 'B12']] = data[['A1', 'A3', 'A4', 'A19', 'B1', 'B12']].fillna(0)

        data['b14/a1_a3_a4_a19_b1_b12'] = data['B14'] / (
                    data['A1'] + data['A3'] + data['A4'] + data['A19'] + data['B1'] + data['B12'])

        data['B12_min_mod_sum_material'] = round(
            data['B12'] / (data['A1'] + data['A3'] + data['A4'] + data['A19'] + data['B1'] + data['B12']), 1)

        self.new_list = self.new_list + ['b14/a1_a3_a4_a19_b1_b12', 'B12_min_mod_sum_material']

    def make_feature3(self,data):
        data['A6_min_mod_B14'] = round((data['A6'] - data['A6'].min()) / data['B14'], 1)
        data['B6_min_mod_B14'] = round((data['B6'] - data['B6'].min()) / data['B14'], 1)
        data['A10_min_mod_B14'] = round((data['A10'] - data['A10'].min()) / data['B14'], 1)
        data['A17_min_mod_B14'] = round((data['A17'] - data['A17'].min()) / data['B14'], 1)
        data['B14_diff_min'] = data['B14'] - data['B14'].min()

        self.new_list = self.new_list + ['A6_min_mod_B14', 'B6_min_mod_B14', 'A10_min_mod_B14', 'A17_min_mod_B14', 'B14_diff_min']

        del data['A1']
        del data['A3']
        del data['A4']
        self.categorical_columns.remove('A1')
        self.categorical_columns.remove('A3')
        self.categorical_columns.remove('A4')

        data[['A19', 'B1', 'B12']] = data[['A19', 'B1', 'B12']].replace(0, -1)


