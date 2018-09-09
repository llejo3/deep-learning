from pandas import Series, DataFrame
from sklearn import preprocessing
import numpy as np
from data.stocks import Stocks


class TrainsData:
    """학습을 위한 데이터를 만든다."""

    def __init__(self, params):
        self.params = params

    @staticmethod
    def to_ndarray(cols_data):
        """matrix 데이터로 변경한다."""
        if isinstance(cols_data, Series):
            return np.reshape(list(cols_data), (-1, 1))
        elif isinstance(cols_data, DataFrame):
            return cols_data.as_matrix()

    def get_scaled_cols(self, data, column_name):
        """컬럼을 스케일링을 시킨다."""
        scale_data = self.to_ndarray(data[column_name])
        scaler = preprocessing.MinMaxScaler()
        return scaler.fit_transform(scale_data), scaler

    def get_scaled_data(self, data):
        """데이터를 스케일링 시킨다."""
        scaled_data = data.copy()
        scaled_data = scaled_data[['close', 'open', 'high', 'low', 'volume']]
        scaled_data = scaled_data[scaled_data['close'] != 0]
        scaled_data['close'], scaler_close = self.get_scaled_cols(scaled_data, 'close')
        scaled_data['open'], _ = self.get_scaled_cols(scaled_data, 'open')
        scaled_data['high'], _ = self.get_scaled_cols(scaled_data, 'high')
        scaled_data['low'], _ = self.get_scaled_cols(scaled_data, 'low')
        scaled_data['volume'], _ = self.get_scaled_cols(scaled_data, 'volume')
        return scaled_data, scaler_close;

    def get_dataXY(self, data):
        """RNN을 위한 데이터로 만든다. """
        x = self.to_ndarray(data)
        y = self.to_ndarray(data['close'])

        dataX = []
        dataY = []
        seq_length = self.params['seq_length']
        y_len = len(y)
        for i in range(0, y_len - seq_length):
            _x = x[i:i + seq_length]
            _y = y[i + seq_length]  # Next close price
            dataX.append(_x)
            dataY.append(_y)
        dataX_last = [x[y_len - seq_length: y_len]]
        return dataX, dataY, y, dataX_last

    def split_train_test(self, dataX, dataY, data, y):
        """train 및 test 데이터로 나눈다."""
        invest_count = self.params['invest_count']
        seq_length = self.params['seq_length']
        data_count = len(dataY)
        train_size = int(data_count * self.params['train_percent'] / 100)
        train_last = data_count - invest_count

        trainX = np.array(dataX[0:train_size])
        testX = np.array(dataX[train_size:train_last])
        investX = np.array(dataX[train_last:data_count])

        trainY = np.array(dataY[0:train_size])
        testY = np.array(dataY[train_size:train_last])
        investY = np.array(dataY[train_last:data_count])

        trainCloses = np.array(y[seq_length - 1:train_size + seq_length - 1])
        testCloses = np.array(dataY[train_size - 1:train_last - 1])
        investCloses = np.array(dataY[train_last - 1:data_count - 1])
        investRealCloses = np.array(data['close'][train_last - 1 + seq_length:data_count + seq_length].values)

        return {
            'trainX': trainX, 'trainY': trainY, 'trainCloses': trainCloses,
            'testX': testX, 'testY': testY, 'testCloses': testCloses,
            'investX': investX, 'investY': investY, 'investCloses': investCloses, 'investRealCloses': investRealCloses
        }

    def get_train_test(self, data):
        """train, test 데이터로 만든다."""
        scaled_data, scaler_close = self.get_scaled_data(data)
        dataX, dataY, y, dataX_last = self.get_dataXY(scaled_data)
        data_params = self.split_train_test(dataX, dataY, data, y)
        return data_params, scaler_close, dataX_last

