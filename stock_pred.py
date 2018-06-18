
# coding: utf-8

# In[ ]:


import zipfile


# In[ ]:


fantasy_zip = zipfile.ZipFile('./Stock_Dataset(2017_07_06).zip')
fantasy_zip.extractall('./data')


# In[1]:


import pandas as pd
from pandas import Series, DataFrame
from sklearn import preprocessing
import numpy as np
import tensorflow as tf
import math


# In[2]:


get_ipython().system('pip install xlrd')


# In[3]:


get_ipython().system('pip install openpyxl')


# In[270]:


### 메소드 정의 
# 상세 데이터를 가져온다.
def get_stock_datail(comp_code) :
    code = format(comp_code, "06d");
    return pd.read_csv('./data/' + code + '.csv')

# matrix 데이터로 변경한다.
def to_ndarray(cols_data) :
    if isinstance(cols_data, Series):
        return np.reshape(list(cols_data), (-1,1))
    elif isinstance(cols_data, DataFrame):
        return cols_data.as_matrix()

# 컬럼을 스케일링을 시킨다.
def get_scaled_cols(data, column_name) :
    scale_data = to_ndarray(data[column_name])
    scaler = preprocessing.MinMaxScaler()
    return scaler.fit_transform(scale_data);

# 데이터를 스케일링 시킨다.
def get_scaled_data(data) :
    scaled_data = data.copy()
    scaled_data['Close'] = get_scaled_cols(scaled_data, 'Close')
    scaled_data['Open'] = get_scaled_cols(scaled_data, 'Open')
    scaled_data['High'] = get_scaled_cols(scaled_data, 'High')
    scaled_data['Low'] = get_scaled_cols(scaled_data, 'Low')
    scaled_data['Volume'] = get_scaled_cols(scaled_data, 'Volume')
    return scaled_data;

# RNN을 위한 데이터로 만든다. 
def get_dataXY(data, train_params) :
    x = to_ndarray(data[['Open', 'High', 'Low', 'Volume', 'Close']])
    y = to_ndarray(data['Close'])
    
    dataX = []
    dataY = []
    seq_length = train_params['seq_length']
    for i in range(0, len(y) - seq_length):
        _x = x[i:i + seq_length]
        _y = y[i + seq_length] # Next close price
        #print(_x, "->", _y)
        dataX.append(_x)
        dataY.append(_y)
    return dataX, dataY

# train 및 test 데이터로 나눈다.
def split_train_test(dataX, dataY, train_params, data) :
    invest_count = train_params['invest_count']
    seq_length = train_params['seq_length']
    data_count = len(dataY);
    train_size = int(data_count * train_params['train_percent'] / 100)
    train_last = data_count-invest_count;
    
    trainX = np.array(dataX[0:train_size])
    testX = np.array(dataX[train_size:train_last])
    investX = np.array(dataX[train_last:data_count])
    
    trainY = np.array(dataY[0:train_size])
    testY = np.array(dataY[train_size:train_last])
    investY = np.array(dataY[train_last:data_count])
    
    #trainCloses = np.array( y[seq_length-1:train_size+seq_length-1])
    testCloses = np.array(dataY[train_size-1:train_last-1])
    investCloses = np.array(dataY[train_last-1:data_count-1])
    investRealCloses = np.array(data['Close'][train_last-1+seq_length:data_count-1+seq_length].values)
    
    return {
        'trainX': trainX, 'trainY': trainY,
        'testX': testX, 'testY': testY, 'testCloses' : testCloses,
        'investX': investX,'investY': investY, 'investCloses': investCloses, 'investRealCloses': investRealCloses
    }

def split_train_test_for_invest(dataX, dataY, train_params, index, data) :
    invest_count = train_params['invest_count']
    seq_length = train_params['seq_length']
    data_count = len(dataY);
    train_size = data_count - invest_count + index
    real_index = train_size + seq_length - 1
    
    trainX = np.array(dataX[0:train_size])
    testX = np.array(dataX[train_size:train_size+1])
    realClose = data['Close'][real_index:real_index+1].values[0]
    
    trainY = np.array(dataY[0:train_size])
    testY = np.array(dataY[train_size:train_size+1])
    
    return {
        'trainX': trainX, 'trainY': trainY, 
        'testX': testX, 'testY': testY,
        'realClose': realClose
    }

# train, test데이터로 만든다.
def get_train_test(data, train_params) :
    scaled_data = get_scaled_data(data)
    dataX, dataY = get_dataXY(scaled_data, train_params)
    return split_train_test(dataX, dataY, train_params, data)

def get_train_test_for_invest(data, train_params, index) :
    scaled_data = get_scaled_data(data)
    dataX, dataY, y = get_dataXY(scaled_data, train_params)
    return split_train_test_for_invest(dataX, dataY, train_params, index, data)

# 텐스플로우 변수관계 그래프롤 그린다.
def draw_graph(train_params) :
    seq_length = train_params['seq_length']
    data_dim = train_params['data_dim']
    
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
    Y = tf.placeholder(tf.float32, [None, 1])
    cell = tf.contrib.rnn.BasicLSTMCell(num_units=train_params['hidden_dim'], 
                                        state_is_tuple=True, 
                                        activation=tf.tanh)
    outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32 )
    Y_pred = tf.contrib.layers.fully_connected(
        outputs[:, -1], train_params['output_dim'], activation_fn=None)  # We use the last cell's output

    # cost/loss
    #closes = tf.placeholder(tf.float32, [None, 1])
    loss = tf.reduce_sum(tf.square(Y_pred - Y))  # sum of the squares
    #loss = tf.losses.mean_squared_error(Y, Y_pred)
    #loss = tf.reduce_sum(-1 * tf.minimum (tf.sign(closes-Y_pred) * tf.sign(closes-Y), 0)) 
    # optimizer
    optimizer = tf.train.AdamOptimizer(train_params['learning_rate'])
    # optimizer = tf.train.RMSPropOptimizer(train_params['learning_rate'])
    train = optimizer.minimize(loss)

    # RMSE
    targets = tf.placeholder(tf.float32, [None, 1])
    predictions = tf.placeholder(tf.float32, [None, 1])
    rmse = tf.sqrt(tf.reduce_mean(tf.square(targets - predictions)))
    
    closes = tf.placeholder(tf.float32, [None, 1])
    direction_success =tf.reduce_mean(tf.maximum(tf.sign(closes-targets) * tf.sign(closes-predictions), 0))
    
    return {
        'X': X,
        'Y': Y,
        'train': train,
        'loss' : loss,
        'Y_pred': Y_pred,
        'targets': targets,
        'rmse' : rmse,
        'predictions': predictions,
        'closes' : closes,
        'direction_success' : direction_success
    }

# 학습을 시킨다.
def let_training(data_params, train_params, graph_params, comp_code) :
    X = graph_params['X']
    Y = graph_params['Y']
    train = graph_params['train']
    loss = graph_params['loss']
    trainX = data_params['trainX']
    trainY = data_params['trainY']
    testX = data_params['testX']
    testY = data_params['testY']
    testCloses = data_params['testCloses']
    
    Y_pred = graph_params['Y_pred']
    targets = graph_params['targets']
    rmse = graph_params['rmse']
    predictions = graph_params['predictions']
    closes = graph_params['closes']
    direction_success = graph_params['direction_success']
    loss_up_cnt = train_params['loss_up_cnt']
    
    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # Training step
        min_rmse_val = 999999
        max_direction_success_val = 0
        less_cnt = 0
        train_count = 0
        for i in range(train_params['iterations']):
            _, step_loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
            if i % 100 == 0 :
                test_predict = sess.run(Y_pred, feed_dict={X: testX})
                rmse_val, direction_success_val = sess.run([rmse,  direction_success], 
                                                feed_dict={targets: testY, predictions: test_predict, closes: testCloses}) 
                #print(step_loss, rmse_val, direction_success_val)
                if rmse_val < min_rmse_val :
                #if direction_success_val > max_direction_success_val :
                    tf.add_to_collection("X", X)
                    tf.add_to_collection("Y", Y)
                    tf.add_to_collection("train", train)
                    tf.add_to_collection("Y_pred", Y_pred)
                    saver.save(sess, "./sessions/" + str(comp_code) + ".ckpt")
                    less_cnt = 0
                    train_count = i;
                    max_test_predict, min_rmse_val, max_direction_success_val = test_predict, rmse_val, direction_success_val
                else :
                    less_cnt += 1
                if less_cnt > loss_up_cnt :
                    break
        
        return max_test_predict, min_rmse_val, max_direction_success_val, train_count 

# 그래프를 그리고 학습을 시킨다.    
def let_leaning(data_params, train_params, comp_code):
    graph_params = draw_graph(train_params)
    return let_training(data_params, train_params, graph_params, comp_code)

def to_dataFrame(data, columns) :
    return pd.DataFrame(data, columns=columns)

# excel로 저장한다.
def save_excel(df_data, file_name):
    writer = pd.ExcelWriter(file_name)
    df_data.to_excel(writer,'Sheet1', index=False)
    writer.save()

# 예측 값에 따라 매수 매도를 실행한다.    
def let_invest_money(invest_predict, now_scaled_close, now_close, rain_params, now_money, now_stock_cnt) :
    seq_length = train_params['seq_length']
    data_dim = train_params['data_dim']
    pie_percent = train_params['pie_percent']
    invest_min_percent = train_params['invest_min_percent']
    
    pie = now_close * pie_percent/100
    ratio = (invest_predict - now_scaled_close) /now_scaled_close * 100
    
    if ratio > invest_min_percent :
        cnt = math.floor(now_money/now_close)
        if cnt > 0 :
            now_money -= (now_close + pie) * cnt
            now_stock_cnt += cnt
    elif ratio < -invest_min_percent :
        if now_stock_cnt > 0 :
            now_money += to_money(now_close, now_stock_cnt, train_params)
            now_stock_cnt = 0
    #print(now_money, now_stock_cnt, now_scaled_close, invest_predict, data_params['testY'])
    return now_money, now_stock_cnt

# 주식매도를 해서 돈으로 바꾼다.
def to_money(now_stock_cnt, now_close, train_params) :
    money = 0
    if now_stock_cnt > 0 :
        pie_percent = train_params['pie_percent'] 
        tax_percent = train_params['tax_percent']
        
        pie = now_close * pie_percent/100
        tax = now_close * tax_percent/100
        money = (now_close - (pie + tax)) * now_stock_cnt
    return money
    
# 학습 후 모의 주식 거래를 한다.
def let_invest(row, train_params, data_params, train_cnt):
    comp_code = row['종목코드']
    invest_count = train_params['invest_count']
    invest_money = train_params['invest_money']
    investX = data_params['investX']
    investCloses = data_params['investCloses']
    investRealCloses = data_params['investRealCloses']
    investX = data_params['investX']
    testX = data_params['testX']
    testY = data_params['testY']
    #print(investRealCloses)
    
    now_stock_cnt = 0
    saver = tf.train.Saver()
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        
        saver.restore(sess, "./sessions/" + str(comp_code) + ".ckpt") 
        X = tf.get_collection('X')[0]
        Y = tf.get_collection('Y')[0]
        train = tf.get_collection('train')[0]
        Y_pred = tf.get_collection('Y_pred')[0]
        
        for i in range(train_cnt):
            sess.run(train, feed_dict={X: testX, Y: testY})
        
        invest_predicts = sess.run(Y_pred, feed_dict={X: investX})
        
        for i in range(0, invest_count) :
            invest_predict = invest_predicts[i][0];
            now_scaled_close = investCloses[i][0]
            now_close = investRealCloses[i]
            #print(invest_predict, now_scaled_close, now_close)
            invest_money, now_stock_cnt = let_invest_money(invest_predict, now_scaled_close, now_close,
                                                           train_params, invest_money, now_stock_cnt)
            #break
        invest_money += to_money(now_stock_cnt, now_close, train_params)
    #print(now_money)
    return invest_money


# In[268]:


# 파라미터 정의 
# train Parameters
train_params = {
    'seq_length' : 7, # 시퀀스 갯수
    'data_dim' : 5,    # 입력 데이터 갯수
    'hidden_dim' : 5,  # 히든 레이어 갯수 
    'output_dim' : 1,  # 출력 데이터 갯수
    'learning_rate' : 0.001, 
    'iterations' : 100000,  # 최대 훈련 반복횟수
    'train_percent' : 70, # 훈련 데이터 퍼센트
    'loss_up_cnt' : 20,
    'invest_corp_count' : 100, # 투자하는 주식회사 갯수
    'invest_count' : 20,  # 투자 횟수
    'invest_money' : 1000000, # 각 주식에 모의투자할 금액
    'pie_percent' : 0.015, # 투자시 발생하는 수수료
    'tax_percent' : 0.5,   # 매도시 발생하는 세금
    'invest_min_percent' : 2.0 # 투자를 하는 최소 간격 퍼센트 
};


# In[ ]:


# 주식회사 데이터
corporations = pd.read_excel('./corporations.xlsx')


# In[ ]:


stock_corps = corporations.query("상장일<'2005-01-01'  ")[['회사명', '종목코드']]
print(stock_corps)


# In[ ]:


# 주식 종목들을 가져와서 학습을 시킨다.
comp_rmses = []
for idx, row in stock_corps.iterrows():
    comp_code = row['종목코드']
    data = get_stock_datail(comp_code)
    data_params = get_train_test(data, train_params)
    _, rmse_val, direction_success_val, train_cnt = let_leaning(data_params, train_params, comp_code)
    
    now_money = let_invest(row, train_params, data_params, train_cnt)
    if idx == 0 :
        print('code', 'name', 'rmse', 'direction_success', 'invest_result')
    print(comp_code, row['회사명'], rmse_val, direction_success_val, now_money)
    comp_rmses.append([comp_code, row['회사명'], rmse_val, direction_success_val, now_money])
    #break
 


# In[ ]:


# 데이터를 정렬하고 저장한다.
df_comp_rmses = pd.DataFrame(comp_rmses, columns=['code', 'name', 'rmse', 'direction_success', 'invest_result'])    
#df_comp_rmses = df_comp_rmses.sort_values('invest_result', ascending=False)
save_excel(df_comp_rmses, 'invest_result.xlsx')

