import numpy as np
import tensorflow as tf
import math
from data.data_utils import DataUtils
from trains.learning import Learning


class MockInvestment:
    """모의투자"""

    def __init__(self, params):
        self.params = params

    def let_invest_money(self, invest_predict, now_scaled_close, now_close, now_money, now_stock_cnt):
        """예측 값에 따라 매수 매도를 실행한다."""
        fee_percent = self.params['fee_percent']
        invest_min_percent = self.params['invest_min_percent']

        ratio = (invest_predict - now_scaled_close) / now_scaled_close * 100

        if ratio > invest_min_percent:
            cnt = math.floor(now_money / now_close)
            if cnt > 0:
                fee = now_close * fee_percent / 100
                now_money -= (now_close + fee) * cnt
                now_stock_cnt += cnt
        elif ratio < -invest_min_percent:
            if now_stock_cnt > 0:
                now_money += self.to_money(now_close, now_stock_cnt)
                now_stock_cnt = 0
        return now_money, now_stock_cnt

    def to_money(self, now_stock_cnt, now_close):
        """주식매도를 해서 돈으로 바꾼다."""
        money = 0
        if now_stock_cnt > 0:
            fee_percent = self.params['fee_percent']
            tax_percent = self.params['tax_percent']

            fee = now_close * fee_percent / 100
            tax = now_close * tax_percent / 100
            money = (now_close - (fee + tax)) * now_stock_cnt
        return money

    def let_invest(self, comp_code, train_cnt, dataX_last, data_params):
        """학습 후 모의 주식 거래를 한다."""
        invest_count = self.params['invest_count']
        invest_money = self.params['invest_money']
        dropout_keep = self.params['dropout_keep']

        # investX = data_params['investX']
        investCloses = data_params['investCloses']
        investRealCloses = data_params['investRealCloses']
        investX = data_params['investX']
        investY = data_params['investY']
        testX = data_params['testX']
        testY = data_params['testY']
        testCloses = data_params['testCloses']

        now_stock_cnt = 0
        saver = tf.train.Saver()
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            file_path = DataUtils.get_session_path(comp_code)
            saver.restore(sess, file_path)
            X = tf.get_collection('X')[0]
            X_closes = tf.get_collection('X_closes')[0]
            Y = tf.get_collection('Y')[0]
            train = tf.get_collection('train')[0]
            Y_pred = tf.get_collection('Y_pred')[0]
            output_keep_prob = tf.get_collection('output_keep_prob')[0]

            for i in range(int(train_cnt)):
                sess.run(train, feed_dict={X: testX, Y: testY, X_closes: testCloses,
                                           output_keep_prob: dropout_keep})

            all_invest_money = invest_money
            all_stock_count = 0
            predicts = []
            now_close = 0
            for i in range(invest_count):
                np.array([1, 2, 3], ndmin=2)
                invest_predicts = sess.run(Y_pred, feed_dict={X: investX[i:i + 1], output_keep_prob: 1.0})
                predicts.append(invest_predicts[0])

                invest_predict = invest_predicts[0][0]
                now_scaled_close = investCloses[0][0]
                now_close = investRealCloses[i]
                # print(invest_predict, now_scaled_close, now_close)
                invest_money, now_stock_cnt = self.let_invest_money(invest_predict, now_scaled_close, now_close,
                                                               invest_money, now_stock_cnt)
                if i==0:
                    all_invest_money, all_stock_count = self.let_invest_money(1.0, now_scaled_close, now_close,
                                                                              all_invest_money, all_stock_count)
                for j in range(int(train_cnt / 5)):
                    sess.run(train,
                             feed_dict={X: investX[j:j + 1], Y: investY[j:j + 1], X_closes: investCloses[j:j + 1],
                                        output_keep_prob: dropout_keep})
                # break
            invest_money += self.to_money(now_stock_cnt, now_close)
            all_invest_money = self.to_money(all_stock_count, now_close)
            graph_params = {'X': X, 'X_closes': X_closes, 'Y': Y, 'train': train,
                            'Y_pred': Y_pred, 'output_keep_prob': output_keep_prob}
            Learning.save_learning_image(sess, saver, graph_params, comp_code)

            last_predict = sess.run(Y_pred, feed_dict={X: dataX_last, output_keep_prob: 1.0})
        # print(now_money)
        return invest_money, last_predict, predicts, all_invest_money
