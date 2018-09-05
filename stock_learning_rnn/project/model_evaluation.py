import pandas as pd
from data.corp import Corp
from data.stocks import Stocks
from data.trains_data import TrainsData
from trains.learning import Learning
from trains.mock_investment import MockInvestment
from data.data_utils import DataUtils


def main():
    corp = Corp()
    corps = corp.get_corps('2004-12-31', ['회사명', '종목코드'])
    stocks = Stocks()
    params = {
        'seq_length': 5,  # 시퀀스 갯수
        'data_dim': 5,  # 입력 데이터 갯수
        'hidden_dims': [128, 96, 64],  # 히든 레이어 갯수
        'dropout_keep': 0.8,  # dropout
        'output_dim': 1,  # 출력 데이터 갯수
        'learning_rate': 0.0001,
        'iterations': [10, 200],  # 최소, 최대 훈련 반복횟수
        'rmse_max': 0.045,
        'train_percent': 80.0,  # 훈련 데이터 퍼센트
        'loss_up_count': 12,  # early stopping
        'invest_count': 50,  # 투자 횟수
        'invest_money': 1000000,  # 각 주식에 모의투자할 금액
        'fee_percent': 0.015,  # 투자시 발생하는 수수료
        'tax_percent': 0.5,  # 매도시 발생하는 세금
        'invest_min_percent': 0.6,  # 투자를 하는 최소 간격 퍼센트
        'kor_font_path': 'C:\\WINDOWS\\Fonts\\H2GTRM.TTF'
    }
    trains_data = TrainsData(params)
    learning = Learning(params)
    invest = MockInvestment(params)
    comp_rmses = []
    no = 1
    for index, corp_data in corps.iterrows():
        corp_code = corp_data['종목코드']
        corp_name = corp_data['회사명']
        stock_data = stocks.get_stock_data(corp_code)
        data_params, scaler_close, dataX_last = trains_data.get_train_test(stock_data)
        rmse_val, train_cnt, rmse_vals, test_predict = learning.let_learning(corp_code, data_params)
        last_money, last_predict, invest_predicts, all_invest_money = invest.let_invest(corp_code, train_cnt, dataX_last, data_params)
        comp_rmses.append([no, corp_code, corp_name, rmse_val, last_money, all_invest_money, train_cnt])
        no += 1

    df_comp_rmses = pd.DataFrame(comp_rmses, columns=['no', 'code', 'name', 'rmse', 'invest_result', 'all_invest_result', 'train_cnt'])
    DataUtils.save_excel(df_comp_rmses, 'training_invest_result.xlsx')


if __name__ == '__main__':
    main()