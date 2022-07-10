import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# #MAE
plt.figure(figsize=(5.5, 5.5))
plt.rc('font',family='Times New Roman')
# 构造x轴刻度标签、数据
label_name = ['House1', 'House2', 'House3', 'House4', 'House5']
Fed_LSTM = [0.010469,0.010632,0.016333,0.010569,0.008831]
Fed_BiLSTM = [0.010588,0.009991,0.014306,0.010364,0.007163]
x = np.arange(len(label_name))  # x轴刻度标签位置
width = 0.25  # 柱子的宽度
plt.bar(x - width/2, Fed_LSTM, width, label='Fed_LSTM')
plt.bar(x + width/2, Fed_BiLSTM, width, label='Fed_BiLSTM')
plt.ylabel('MAE')
plt.xticks(np.arange(5),('Household1', 'Household2', 'Household3', 'Household4', 'Household5'))
plt.legend(loc='upper left')
plt.savefig('MAE.png',dpi=500,bbox_inches='tight')


#MAPE
plt.figure(figsize=(5.5, 5.5))
plt.rc('font',family='Times New Roman')
# 构造x轴刻度标签、数据
labels = ['House1', 'House2', 'House3', 'House4', 'House5']
Fed_LSTM = [17.4744,16.9805,20.507,22.971,9.995]
Fed_BiLSTM = [14.0436,13.0841,16.2651,20.3177,7.0755]
x = np.arange(len(label_name))  # x轴刻度标签位置
width = 0.25  # 柱子的宽度
plt.bar(x - width/2, Fed_LSTM, width, label='Fed_LSTM')
plt.bar(x + width/2, Fed_BiLSTM, width, label='Fed_BiLSTM')
plt.ylabel('MAPE')
plt.xticks(np.arange(5),('Household1', 'Household2', 'Household3', 'Household4', 'Household5'))
plt.legend()
plt.savefig('MAPE.png',dpi=500,bbox_inches='tight')

#RMSE
plt.figure(figsize=(5.5, 5.5))
plt.rc('font',family='Times New Roman')
# 构造x轴刻度标签、数据
labels = ['House1', 'House2', 'House3', 'House4', 'House5']
Fed_LSTM = [0.023828,0.025773,0.036922,0.023355,0.024142]
Fed_BiLSTM = [0.023696,0.025626,0.035545,0.022758,0.023041]
plt.figure(figsize=(5.5, 5.5))
plt.rc('font',family='Times New Roman')
# 构造x轴刻度标签、数据
x = np.arange(len(label_name))  # x轴刻度标签位置
width = 0.25  # 柱子的宽度
plt.bar(x - width/2, Fed_LSTM, width, label='Fed_LSTM')
plt.bar(x + width/2, Fed_BiLSTM, width, label='Fed_BiLSTM')
plt.ylabel('RMSE')
plt.xticks(np.arange(5),('Household1', 'Household2', 'Household3', 'Household4', 'Household5'))
plt.legend()
plt.savefig('RMSE.png',dpi=500,bbox_inches='tight')


#
# # # 两组数据
# # plt.subplot(131)
# x = np.arange(len(labels))  # x轴刻度标签位置
# width = 0.25  # 柱子的宽度
# # 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
# # x - width/2，x + width/2即每组数据在x轴上的位置
# plt.bar(x - width/2, Fed_LSTM, width, label='Fed_LSTM')
# plt.bar(x + width/2, Fed_BiLSTM, width, label='Fed_BiLSTM')
# plt.ylabel('MAE')
# # plt.title('2 datasets')
# # x轴刻度标签位置不进行计算
# plt.xticks(x, labels=labels)
# plt.legend()
# # # 三组数据
# # plt.subplot(132)
# # x = np.arange(len(labels))  # x轴刻度标签位置
# # width = 0.25  # 柱子的宽度
# # # 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
# # # x - width，x， x + width即每组数据在x轴上的位置
# # plt.bar(x - width, first, width, label='1')
# # plt.bar(x, second, width, label='2')
# # plt.bar(x + width, third, width, label='3')
# # plt.ylabel('Scores')
# # plt.title('3 datasets')
# # # x轴刻度标签位置不进行计算
# # plt.xticks(x, labels=labels)
# # plt.legend()
# # 四组数据
# # plt.subplot(133)
# # x = np.arange(len(labels))  # x轴刻度标签位置
# # width = 0.2  # 柱子的宽度
# # # 计算每个柱子在x轴上的位置，保证x轴刻度标签居中
# # plt.bar(x - 1.5*width, first, width, label='LSTM')
# # plt.bar(x - 0.5*width, second, width, label='BiLSTM')
# # plt.bar(x + 0.5*width, third, width, label='Fed_LSTM')
# # plt.bar(x + 1.5*width, fourth, width, label='Fed_BiLSTM')
# # plt.ylabel('Error')
# # plt.xlabel('Households')
# # plt.title('RMSE')
# # x轴刻度标签位置不进行计算
# plt.xticks(x, labels=labels)
# plt.legend()
# plt.savefig('figure3.png',dpi=500,bbox_inches='tight')
# # plt.show()
