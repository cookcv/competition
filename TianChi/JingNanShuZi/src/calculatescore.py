import pandas as pd


def calculateScore(test_data_path,predictions,ans_data_path):

    # ans_data_path_B = '../../data/jinnan_round1_ansB_20190125.csv'
    # ans_data_path_A = '../../data/jinnan_round1_ansA_20190125.csv'
    # ans_data_path_C = '../../data/jinnan_round1_ans_20190201.csv'

    ans_data = pd.read_csv(ans_data_path, header=None)
    test_df = pd.read_csv(test_data_path,encoding='gb2312')
    sub_df = pd.DataFrame()
    sub_df['样本id'] = test_df['样本id']
    sub_df[1] = predictions

    print(sum((ans_data[1]-sub_df[1])**2/(sub_df.shape[0]*2)))