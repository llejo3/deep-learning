import pandas as pd
from data.corp import Corp
from data.stocks import Stocks
from data.trains_data import TrainsData
from trains.learning import Learning
from trains.mock_investment import MockInvestment
from data.data_utils import DataUtils


def let_train_invest(corp_code, corp_name, params, no):
    stocks = Stocks()
    trains_data = TrainsData(params)
    learning = Learning(params)
    invest = MockInvestment(params)

    stock_data = stocks.get_stock_data(corp_code)
    data_params, scaler_close, dataX_last = trains_data.get_train_test(stock_data)
    rmse_val, train_cnt, rmse_vals, test_predict = learning.let_learning(corp_code, data_params)
    last_money, last_predict, invest_predicts, all_invest_money = invest.let_invest(corp_code, train_cnt, dataX_last,
                                                                                    data_params)
    learning.draw_plot(rmse_vals, test_predict, invest_predicts, corp_name, data_params)
    last_close_money, last_pred_money = invest.get_real_money(data_params, scaler_close, last_predict)
    print("RMSE:", rmse_val)
    print("train_cnt:", train_cnt)
    if params['invest_count'] > 0:
        print(str(params['invest_count']) + "회 모의투자 결과(100만원 투자):", "{:,.0f}".format(last_money))
    last_date = stock_data.tail(1)['date'].to_string(index=False)
    print("마지막 종가(" + last_date + "):", "{:,.0f}".format(last_close_money))
    last_pred_ratio = (last_pred_money - last_close_money) / last_close_money * 100
    last_pred_ratio = "(" + "{:.2f}".format(last_pred_ratio) + "%)"
    print("예측 종가:", "{:,.0f}".format(last_pred_money), last_pred_ratio)
    print()
    return [no, last_date, corp_code, corp_name, rmse_val, train_cnt, last_close_money, last_pred_money, last_pred_ratio]


def let_train_invests(corp_names, params):
    corp = Corp()
    comp_rmses = []
    no = 1
    for corp_name in corp_names:
        corp_code = corp.get_comp_code(corp_name)
        result = let_train_invest(corp_code, corp_name, params, no)
        comp_rmses.append(result)
        no += 1
    df_comp_rmses = pd.DataFrame(comp_rmses,
                                 columns=['no', 'last_date', 'code', 'name', 'rmse', 'train_cnt', 'last_close_money',
                                          'last_pred_money', 'last_pred_ratio'])
    DataUtils.save_excel(df_comp_rmses, './result/forcast_result.xlsx')

def main(corp_names = ["삼성중공업","기아자동차", "게임빌","루트로닉", "영진약품", "대아티아이"]):
    params = {
        'seq_length': 5,  # 시퀀스 갯수
        'data_dim': 5,  # 입력 데이터 갯수
        'hidden_dims': [128, 96, 64],  # 히든 레이어 갯수
        'dropout_keep': 0.8,  # dropout
        'output_dim': 1,  # 출력 데이터 갯수
        'learning_rate': 0.0001,
        'iterations': [1000, 10000],  # 최소, 최대 훈련 반복횟수
        'rmse_max': 0.02,
        'train_percent': 80.0,  # 훈련 데이터 퍼센트
        'loss_up_count': 100,  # early stopping
        'invest_count': 0,  # 투자 횟수
        'invest_money': 10000000,  # 각 주식에 모의투자할 금액
        'fee_percent': 0.015,  # 투자시 발생하는 수수료
        'tax_percent': 0.5,  # 매도시 발생하는 세금
        'invest_min_percent': 0.6,  # 투자를 하는 최소 간격 퍼센트
        'kor_font_path': 'C:\\WINDOWS\\Fonts\\H2GTRM.TTF'
    }
    let_train_invests(corp_names, params)


if __name__ == '__main__':
    main()
