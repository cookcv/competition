import pandas as pd
import numpy as np
import re


class PrepareData:

    def __init__(self,train_data,test_data):
        # 变量初始化
        self.train = train_data
        self.test = test_data


    def exceptionHandling(self):
        good_cols = list(self.train.columns)
        self.train = self.train[self.train['收率'] > 0.87]
        self.train.loc[self.train['B14'] == 40,'样本id'] = 400
        self.train = self.train[self.train['B14'] > 350]
        self.train = self.train[self.train['B8'] < 55]
        #     train = train[train['B1']>180]
        self.train.loc[self.train['样本id'] == 'sample_1590', 'A25'] = 0
        # train = train[train['A25']>60]
        self.train['A25'] = self.train['A25'].apply(lambda x: int(x))
        self.train = self.train[self.train['A25'] > 60]
        self.train = self.train[self.train['A22'] > 7]
        self.train = self.train[self.train['A21'] < 80]
        self.train = self.train[self.train['A17'] >= 100]
        self.train = self.train[self.train['B13'] >= 0.06]
        self.train = self.train[self.train['A23'] == 5]
        self.train = self.train[self.train['A18'] == 0.2]
        self.train = self.train[self.train['A13'] == 0.2]
        #     train = train[train['B3']==3.5]
        self.train = self.train[self.train['A3'] > 350]
        self.train = self.train[self.train['A1'] > 240]
        self.train = self.train[self.train['A6'] < 70]  # [<50]

        self.train.reset_index(drop=True, inplace=True)
        self.train = self.train[good_cols]
        good_cols.remove('收率')

        self.test = self.test[good_cols]
        self.test['A3'] = 405

        target = self.train['收率']
        del self.train['收率']
        data = pd.concat([self.train, self.test], axis=0, ignore_index=True)
        data.drop(['B3', 'B13', 'A8', 'A13', 'A18', 'A23'], axis=1, inplace=True)

        data['样本id'] = data['样本id'].apply(lambda x: int(x.split('_')[1]))

        # 将时间点转换
        time_colums_list = ['A5', 'A7', 'A9', 'A11', 'A14', 'A16', 'A24', 'A26', 'B5', 'B7']
        for f in time_colums_list:
            data[f] = data[f].apply(self.timeTranSecond)
        # 将时间段转换
        for f in ['A20', 'A28', 'B4', 'B9', 'B10', 'B11']:
            data[f] = data.apply(lambda df: self.getDuration(df[f]), axis=1)

        return data,target,self.train,self.test

    # 将时间点转换
    def timeTranSecond(self,t):
        if pd.isna(t):
            return np.NaN

        if t == '1900/1/9 7:00':
            return 7 * 3600 / 3600
        elif t == '1900/1/1 2:30':
            return (2 * 3600 + 30 * 60) / 3600
        elif t == -1:
            return -1
        else:
            t = str(t).replace('::', ':').replace('；', ':').replace(';', ':').replace('分', '').replace('"', ':')

        if t == '1600':
            t = '16:00'

        t_list = t.split(':')
        if len(t_list) == 3:
            t, m, s = t_list
        elif len(t_list) == 2:
            t, m = t_list
            s = 0
        else:
            print(t)

        try:
            tm = (int(t) * 3600 + int(m) * 60 + int(s)) / 3600
        except:
            return (30 * 60) / 3600

        return tm

    # 将时间段转换
    def getDuration(self,se):
        tm = np.NaN
        if se == -1:
            return -1
        elif pd.isna(se):
            return -1
        elif se == '19:-20:05':
            se = '19:00-20:05'
        elif se == '15:00-1600':
            se = '15:00-16:00'

        sh, sm, eh, em = re.findall(r"\d+\.?\d*", se)

        try:
            if int(sh) > int(eh):
                tm = (int(eh) * 3600 + int(em) * 60 - int(sm) * 60 - int(sh) * 3600) / 3600 + 24
            else:
                tm = (int(eh) * 3600 + int(em) * 60 - int(sm) * 60 - int(sh) * 3600) / 3600
        except:
            return -1
        return tm