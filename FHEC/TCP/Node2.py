# -*- coding: UTF-8 -*-

import pickle
import re
import socket
import time
import torch
from matplotlib.animation import FuncAnimation
from torch import nn
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
import numpy as np
# from causality.Model import LSTM

look_back = 8
EPOCH = 16
head = [None for i in range(look_back)]
SIZE = 12*700

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
scaler = MinMaxScaler(feature_range=(0, 1))

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)

def mape_value(real, pred):
    return numpy.abs(real-pred)/real

def log(info):
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' + str(info))

# 一、数据预处理
# load the dataset
def fl_data_load(path):
    dataframe = read_csv(path, engine='python')
    dataset = dataframe[0:9600].values
    dataset = dataset.astype('float32')
    numpy.random.seed(7)
    dataset = scaler.fit_transform(dataset)
    data_X, data_Y = create_dataset(dataset, look_back)

    train_size = SIZE - look_back
    test_size = len(dataset) - train_size
    train_X = data_X[:train_size]
    train_Y = data_Y[:train_size]
    test_X = data_X[train_size:]
    test_Y = data_Y[train_size:]

    train_X = numpy.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
    train_Y = train_Y.reshape(-1, 1, 1)
    test_X = numpy.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))

    var_x = torch.from_numpy(train_X)
    var_y = torch.from_numpy(train_Y)
    var_testX = torch.from_numpy(test_X)

    return var_x, var_y, var_testX, test_Y

path = r"C:\Users\Sun\PycharmProjects\FL-master\dataset\house2_5min_KWh.csv"

var_x, var_y, var_testX, test_Y = fl_data_load(path)

class LSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=4, output_size=1, num_layer=1,bidirectional=True):
        super(LSTM, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer,bidirectional)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.layer1(x)
        x = torch.relu(x)
        s, b, h = x.size()
        x = x.view(s * b, h)
        x = self.layer2(x)
        x = x.view(s, b, -1)
        return x


model = LSTM(look_back, 4, 1, 2)
loss_fun = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

losses = list()
steps = list()
for epoch in range(1, EPOCH + 1):
    log("\033[1;31;40m第\033[1;31;40m%s\033[1;31;40m轮开始训练!\033[1;31;40m" % str(epoch))

    for t in range(10):
        loss_t = list()

        out = model(var_x)
        loss = loss_fun(out, var_y)
        loss_t.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses.append(sum(loss_t)/len(loss_t))
    steps.append(epoch)
    plt.plot(steps, losses, "o-")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.draw()
    plt.pause(0.1)


    log("建立连接并上传......")
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    host = '192.168.1.102'
    port = 6138
    s.connect((host, port))
    data = {}
    data['num'] = epoch
    data['model'] = model.state_dict()

    keys = model.state_dict().keys()


    data = pickle.dumps(data)
    print(s.send(data))
    log("等待接收......")
    try:
        s.settimeout(30000)
        data = s.recv(1024 * 100)
        # print(data)
        data = pickle.loads(data)
        print(data['num'], epoch)
        if data['num'] == epoch:
            global_state_dict = data['model']
        else:
            global_state_dict = model.state_dict()
    except Exception as e:
        print(e)
        # s.sendto(data, (host, port))
        log("没有在规定时间收到正确的包， 利用本地参数更新")
        global_state_dict = model.state_dict()

    model.load_state_dict(global_state_dict)
    s.close()
log("训练完毕，关闭连接")
s.close()
plt.close()
with torch.no_grad():
    model.eval()
    pred_testY = model(var_testX)
    pred_testY = pred_testY.view(-1, 1).data.numpy()
    test_Y_origin = scaler.inverse_transform(test_Y.reshape(-1, 1))
    pred_testY_origin = scaler.inverse_transform(pred_testY)
    mape = numpy.array(list(map(lambda x: x if x < 1 else 0, mape_value(test_Y_origin, pred_testY_origin)))).mean()
    MAE = metrics.mean_absolute_error(np.array(test_Y_origin), np.array(pred_testY_origin))
    MSE = metrics.mean_squared_error(np.array(test_Y_origin), np.array(pred_testY_origin))
    RMSE = np.sqrt(metrics.mean_squared_error(np.array(test_Y_origin), np.array(pred_testY_origin)))
    R2 = metrics.r2_score(np.array(test_Y_origin), np.array(pred_testY_origin))
    Acc = metrics.explained_variance_score(np.array(test_Y_origin), np.array(pred_testY_origin))

    plt.ioff()
    plt.figure(1, figsize=(8, 5))
    plt.rc('font', family='Times New Roman')
    plt.clf()
    # plt.grid(True, linestyle='--')
    plt.plot(test_Y_origin, label='Real Value',linewidth=1)  # color='#65A1D7'
    plt.xlim(100,200)
    plt.plot(pred_testY_origin, label='Predicted Value',linewidth=1)  # color='#ED7D31'
    plt.title("House 2",fontsize=22)
    plt.legend(loc='best')
    plt.xlabel('Time', fontsize=18)
    plt.ylabel('HEC', fontsize=18)
    plt.savefig('Node2.png',dpi=500,bbox_inches='tight')
    plt.show()

    print("MAPE：{:.6f}".format(mape[0]))
    print("MAE：{:.6f}".format(MAE))
    print("MSE：{:.6f}".format(MSE))
    print("RMSE：{:.6f}".format(RMSE))
    print("R2：{:.6f}".format(R2))
    print("Acc：{:.6f}".format(Acc))


    data = pd.DataFrame({'time': range(0, len(test_Y_origin)),
                         'r': test_Y_origin[:, 0],
                         'p': pred_testY_origin[:, 0]})
    fig, ax = plt.subplots()
    # def update(curr_time):
    #     data_test = data.loc[data.time <= curr_time, :]
    #     idx = data_test.time
    #     ax.clear()
    #     # r
    #     ax.plot(idx, data_test['r'], color='#FF5872', lw=4)  # 折线图
    #     ax.scatter(idx.tolist()[-1], data_test['r'].tolist()[-1], color='#FF5872',
    #                edgecolor='black', s=280, lw=2.5, zorder=4)  # 散点图
    #     ax.text(idx.tolist()[-1] + 5, data_test['r'].tolist()[-1], 'result',
    #             size=15, color='#FF5872', va='top', ha='left', fontweight='bold')
    #     ax.text(idx.tolist()[-1] + 23, data_test['r'].tolist()[-1], '   ={: ,.0f}  '.format(data_test['r'].tolist()[-1]),
    #             size=15, color='#FF5872', va='top', ha='left', fontweight='bold')
    #
    #     # p
    #     ax.plot(idx, data_test['p'], color='#7FEB00', lw=2)  # 折线图
    #     ax.scatter(idx.tolist()[-1], data_test['p'].tolist()[-1], color='#7FEB00',
    #                edgecolor='black', s=280, lw=2.5, zorder=4)  # 散点图
    #     ax.text(idx.tolist()[-1] + 52, data_test['p'].tolist()[-1], ' predict',
    #             size=15, color='#7FEB00', va='top', ha='left', fontweight='bold')
    #     ax.text(idx.tolist()[-1] + 76, data_test['p'].tolist()[-1], '    ={: ,.0f}'.format(data_test['p'].tolist()[-1]),
    #             size=15, color='#7FEB00', va='top', ha='left', fontweight='bold')
    #
    #     ax.axvline(idx.tolist()[-1], ls="--", color='gray', lw=1)
    #
    #     ax.spines['top'].set_visible(False)
    #     ax.spines['right'].set_visible(False)
    #     ax.spines['left'].set_color('#373E4B')
    #     ax.spines['bottom'].set_color('#373E4B')
    #     ax.tick_params(bottom=False, size=15, direction='in', colors='gray')
    #     ax.set_xticks(numpy.arange(0, len(test_Y_origin)+100, 20))
    #
    #     ax.set_xlim(left=0, right=len(test_Y_origin)+100)
    #     ax.set_ylim(ymin=min(test_Y_origin)[0]-100, ymax=max(test_Y_origin)[0]+100)
    #     ax.set_xlabel("Time", color="#425663", fontsize=12)
    #     ax.set_ylabel("Traffic", color="#425663", fontsize=12)
    #     ax.set_title(re.split("[/,\,.]", str(__file__))[-2], color="#425663", fontsize=17)
    #
    #     ax.grid(axis="both", color='gray', lw=1, alpha=.6, ls='--')
    #     ax.set_axisbelow(True)
    #     plt.box(True)
    #
    #     time.sleep(0.2)
    #
    # ani = FuncAnimation(fig, update, frames=range(0, len(data)), interval=20)
    # plt.show()
    # plt.close()
