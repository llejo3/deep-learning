import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl
from data.data_utils import DataUtils


class Learning:
    """학습을 시킨다"""

    def __init__(self, params):
        self.params = params

    def draw_graph(self):
        """텐스플로우 변수관계 그래프롤 그린다."""
        seq_length = self.params['seq_length']
        data_dim = self.params['data_dim']
        hidden_dims = self.params['hidden_dims']

        tf.reset_default_graph()
        X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
        X_closes = tf.placeholder(tf.float32, [None, 1])
        Y = tf.placeholder(tf.float32, [None, 1])
        output_keep_prob = tf.placeholder(tf.float32)

        cells = []
        for n in hidden_dims:
            cell = tf.contrib.rnn.BasicLSTMCell(num_units=n, activation=tf.tanh)
            dropout_cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=output_keep_prob)
            cells.append(dropout_cell)
        stacked_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        outputs, _states = tf.nn.dynamic_rnn(stacked_rnn_cell, X, dtype=tf.float32)
        Y_pred = tf.contrib.layers.fully_connected(
            outputs[:, -1], self.params['output_dim'], activation_fn=None)  # We use the last cell's output

        # cost/loss
        loss = tf.reduce_sum(tf.square(1 - (1 + Y_pred - X_closes) / (1 + Y - X_closes)))

        optimizer = tf.train.AdamOptimizer(self.params['learning_rate'])
        train = optimizer.minimize(loss)

        # RMSE
        targets = tf.placeholder(tf.float32, [None, 1])
        predictions = tf.placeholder(tf.float32, [None, 1])
        rmse = tf.sqrt(tf.reduce_mean(tf.square(1 - (1 + predictions - X_closes) / (1 + targets - X_closes))))

        return {
            'X': X,
            'Y': Y,
            'output_keep_prob': output_keep_prob,
            'train': train,
            'loss': loss,
            'Y_pred': Y_pred,
            'targets': targets,
            'rmse': rmse,
            'predictions': predictions,
            'X_closes': X_closes
        }

    def draw_plot(self, rmse_vals, test_predict, invest_predicts, comp_name, data_params):
        testY = data_params['testY']
        investY = data_params['investY']
        y = np.append(testY, investY)
        predict = np.append(test_predict, invest_predicts)

        mpl.rcParams['axes.unicode_minus'] = False
        font_name = fm.FontProperties(fname=self.params['kor_font_path'], size=50).get_name()
        plt.rc('font', family=font_name)

        plt.figure(1)
        plt.plot(rmse_vals, 'gold')
        plt.xlabel('Epoch')
        plt.ylabel('Root Mean Square Error')
        plt.title(comp_name)

        plt.figure(2)
        plt.plot(y, 'b')
        plt.plot(predict, 'r')
        plt.xlabel('Time Period')
        plt.ylabel('Stock Price')
        plt.title(comp_name)
        plt.show()

    @staticmethod
    def save_learning_image(sess, saver, comp_code):
        file_path = DataUtils.get_session_path(comp_code)
        saver.save(sess, file_path)

    def let_training(self, graph_params, comp_code, data_params):
        """학습을 시킨다."""
        X = graph_params['X']
        Y = graph_params['Y']
        output_keep_prob = graph_params['output_keep_prob']
        train = graph_params['train']
        loss = graph_params['loss']
        trainX = data_params['trainX']
        trainY = data_params['trainY']
        testX = data_params['testX']
        testY = data_params['testY']
        trainCloses = data_params['trainCloses']
        testCloses = data_params['testCloses']

        Y_pred = graph_params['Y_pred']
        targets = graph_params['targets']
        rmse = graph_params['rmse']
        predictions = graph_params['predictions']
        X_closes = graph_params['X_closes']
        loss_up_count = self.params['loss_up_count']
        dropout_keep = self.params['dropout_keep']
        iterations = self.params['iterations']
        rmse_max = self.params['rmse_max']

        saver = tf.train.Saver()
        session_path = DataUtils.get_session_path(comp_code)
        restored = False

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            if os.path.isfile(session_path + '.index'):
                saver.restore(sess, session_path)
                iterations[0] = 0
                restored = True

            # Training step
            min_rmse_val = 999999
            less_cnt = 0
            train_count = 0
            rmse_vals = []

            for i in range(iterations[1]):
                if not restored or i != 0:
                    _, step_loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY, X_closes: trainCloses,
                                                                      output_keep_prob: dropout_keep})
                test_predict = sess.run(Y_pred, feed_dict={X: testX, output_keep_prob: 1.0})
                rmse_val = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict, X_closes: testCloses})
                rmse_vals.append(rmse_val)

                if i == 0 and restored:
                    max_test_predict, min_rmse_val, = test_predict, rmse_val

                if rmse_val < min_rmse_val:
                    self.save_learning_image(sess, saver, comp_code)
                    less_cnt = 0
                    train_count = i;
                    max_test_predict, min_rmse_val, = test_predict, rmse_val
                else:
                    less_cnt += 1
                if i >= iterations[0] and less_cnt > loss_up_count and rmse_max > min_rmse_val:
                    break
            # draw_plot(rmse_vals, max_test_predict, testY, comp_name)
            return min_rmse_val, train_count, rmse_vals, max_test_predict

    def let_learning(self, comp_code, data_params):
        """그래프를 그리고 학습을 시킨다."""
        graph_params = self.draw_graph()
        return self.let_training(graph_params, comp_code, data_params)