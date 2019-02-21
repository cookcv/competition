import pandas as pd
import argparse
import warnings
from exceptionhandling import PrepareData
from makefeature import MakeFeature
from model import lgbModel, xgbModel, modelResultMerge

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns',None)
pd.set_option('max_colwidth',100)


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training JinNan.')
    parser.add_argument('--train_path',help='The path of train data',default='../data/jinnan_round1_train_20181227.csv')
    parser.add_argument('--test_path',help='The path of test data',default='../data/jinnan_round1_testA_20181227.csv')
    parser.add_argument('--test_ans_path',help='The path of test data',default='../data/jinnan_round1_ansA_20190125.csv')
    parser = parser.parse_args(args)

    # set data path
    train_data_path = parser.train_path
    test_data_path = parser.test_path

    # get data
    train_data = pd.read_csv(train_data_path, encoding='gb18030')
    test_data = pd.read_csv(test_data_path, encoding='gb18030')

    data,target,train_data_2,test_data_2 = PrepareData(train_data,test_data).exceptionHandling()
    train_length = train_data_2.shape[0]
    test_length = test_data_2.shape[0]
    # make data
    X_train,X_test = MakeFeature(data,train_length,target).makeFeature()
    Y_train = target.values

    # make model to train
    predictions_lgb,train_lgb = lgbModel(train_length,test_length,X_train,Y_train,X_test)
    predictions_xgb,train_xgb = xgbModel(train_length,test_length,X_train,Y_train,X_test)
    # merge result
    predictions = modelResultMerge(predictions_lgb,predictions_xgb,train_lgb,train_xgb,target)

    sub_df = pd.DataFrame()
    sub_df['样本id'] = test_data['样本id']
    sub_df[1] = predictions

    #sub_df.to_csv(root_data_path+"../submit/sub_jinnan_1113_merge2_%s.csv"%(datetime.datetime.now().strftime('%Y%m%d_%H%M%S')), index=False, header=None)
    # 根据提供的答案计算得分
    ans_data = pd.read_csv(parser.test_ans_path, header=None)
    print(sum((ans_data[1]-sub_df[1])**2/(sub_df.shape[0]*2)))

if __name__ == '__main__':
    main()