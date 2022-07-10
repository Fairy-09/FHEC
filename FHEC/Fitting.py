import numpy as np
import matplotlib.pyplot as plt

s_linestyle='-'
s_marker=''



dict_cdir1 = 'saved\\file' + 'Node1.npy'
dict_load = np.load(dict_cdir1, allow_pickle=True).item()
Node1_FedBiLSTM_pre = dict_load['Node1_FedBiLSTM_pre']
Node1_FedBiLSTM_real = dict_load['Node1_FedBiLSTM_real']
dict_cdir11 = 'saved\\file' + 'Node1LSTM.npy'
dict_load11 = np.load(dict_cdir11, allow_pickle=True).item()
Node1_FedLSTM_pre = dict_load11['Node1_FedLSTM_pre']
dict_cdir111='saved\\fileNode1SingleLSTM.npy'
dict_load111=np.load(dict_cdir111, allow_pickle=True).item()
Node1_LSTM_pre = dict_load111['Node1_LSTM_pre']
dict_cdir1111='saved\\fileNode1SingleBiLSTM.npy'
dict_load1111=np.load(dict_cdir1111, allow_pickle=True).item()
Node1_BiLSTM_pre = dict_load1111['Node1_BiLSTM_pre']
plt.ioff()
plt.figure(1, figsize=(15, 2))
plt.rc('font', family='Times New Roman')
plt.clf()
# plt.grid(True, linestyle='--')
plt.plot(Node1_FedBiLSTM_real, label='Real Value', linewidth=1.5,marker='s',markersize=2)  # color = '#65A1D7'
plt.xlim(400, 1100)
plt.rc('font', family='Times New Roman', size=16)
plt.plot(Node1_LSTM_pre, label='LSTM', linewidth=1,linestyle=s_linestyle)
plt.plot(Node1_BiLSTM_pre, label='BiLSTM', linewidth=1,linestyle=s_linestyle)
plt.plot(Node1_FedLSTM_pre, label='Fed_LSTM', linewidth=1,linestyle=s_linestyle,color='#B9D4DB')  # color='#ED7D31'
plt.plot(Node1_FedBiLSTM_pre, label='Fed_BiLSTM', linewidth=1,linestyle=s_linestyle,color='#DE5B6D')
# plt.subplots_adjust(left=0.05, bottom=0.125, right=0.99, top=0.99)
# plt.title("House 1", fontsize=20)
plt.legend(loc='best',fontsize=10)
# plt.xlabel('Time', fontsize=18)
plt.ylabel('House 1', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('Node1.png',dpi=500,bbox_inches='tight')   #去掉边缘空白








dict_cdir2 = 'saved\\file' + 'Node2.npy'
dict_load = np.load(dict_cdir2, allow_pickle=True).item()
Node2_FedBiLSTM_pre = dict_load['Node2_FedBiLSTM_pre']
Node2_FedBiLSTM_real = dict_load['Node2_FedBiLSTM_real']
dict_cdir22 = 'saved\\file' + 'Node2LSTM.npy'
dict_load22 = np.load(dict_cdir22, allow_pickle=True).item()
Node2_FedLSTM_pre = dict_load22['Node2_FedLSTM_pre']
dict_cdir222='saved\\fileNode2SingleLSTM.npy'
dict_load222=np.load(dict_cdir222, allow_pickle=True).item()
Node2_LSTM_pre = dict_load222['Node2_LSTM_pre']
dict_cdir2222='saved\\fileNode2SingleBiLSTM.npy'
dict_load2222=np.load(dict_cdir2222, allow_pickle=True).item()
Node2_BiLSTM_pre = dict_load2222['Node2_BiLSTM_pre']
plt.ioff()
plt.figure(1, figsize=(15,2))
plt.rc('font', family='Times New Roman')
plt.clf()
# plt.grid(True, linestyle='--')
plt.plot(Node2_FedBiLSTM_real, label='Real Value', linewidth=1.5,marker=s_marker)  # color='#65A1D7'
plt.xlim(400, 1100)
plt.plot(Node2_LSTM_pre, label='LSTM', linewidth=1,linestyle=s_linestyle)
plt.plot(Node2_BiLSTM_pre, label='BiLSTM', linewidth=1,linestyle=s_linestyle)
plt.plot(Node2_FedLSTM_pre, label='Fed_LSTM', linewidth=1,linestyle=s_linestyle,color='#B9D4DB')
plt.plot(Node2_FedBiLSTM_pre, label='Fed_BiLSTM', linewidth=1,linestyle=s_linestyle,color='#DE5B6D')  # color='#ED7D31'
# plt.title("House 2", fontsize=20)
# plt.legend(loc='best')
# plt.xlabel('Time', fontsize=18)
# plt.legend(loc='best',fontsize=10)
plt.ylabel('House 2', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('Node2.png',dpi=500,bbox_inches='tight')
# plt.show()
















dict_cdir3 = 'saved\\file' + 'Node3.npy'
dict_load = np.load(dict_cdir3, allow_pickle=True).item()
Node3_FedBiLSTM_pre = dict_load['Node3_FedBiLSTM_pre']
Node3_FedBiLSTM_real = dict_load['Node3_FedBiLSTM_real']
dict_cdir33 = 'saved\\file' + 'Node3LSTM.npy'
dict_load33 = np.load(dict_cdir33, allow_pickle=True).item()
Node3_FedLSTM_pre = dict_load33['Node3_FedLSTM_pre']
dict_cdir333='saved\\fileNode3SingleLSTM.npy'
dict_load333=np.load(dict_cdir333, allow_pickle=True).item()
Node3_LSTM_pre = dict_load333['Node3_LSTM_pre']
dict_cdir3333='saved\\fileNode3SingleBiLSTM.npy'
dict_load3333=np.load(dict_cdir3333, allow_pickle=True).item()
Node3_BiLSTM_pre = dict_load3333['Node3_BiLSTM_pre']
plt.ioff()
plt.figure(1, figsize=(15, 2))
plt.rc('font', family='Times New Roman')
plt.clf()
# plt.grid(True, linestyle='--')
plt.plot(Node3_FedBiLSTM_real, label='Real Value', linewidth=1.5,marker=s_marker)  # color='#65A1D7'
plt.xlim(400, 1100)
plt.plot(Node3_LSTM_pre, label='LSTM', linewidth=1,linestyle=s_linestyle)
plt.plot(Node3_BiLSTM_pre, label='BiLSTM', linewidth=1,linestyle=s_linestyle)
plt.plot(Node3_FedLSTM_pre, label='Fed_LSTM', linewidth=1,linestyle=s_linestyle,color='#B9D4DB')  # color='#ED7D31'
plt.plot(Node3_FedBiLSTM_pre, label='Fed_BiLSTM', linewidth=1,linestyle=s_linestyle,color='#DE5B6D')
# plt.title("House 3", fontsize=20)
# plt.legend(loc='best')
# plt.xlabel('Time', fontsize=18)
plt.ylabel('House 3', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('Node3.png',dpi=500,bbox_inches='tight')
# plt.show()











dict_cdir4 = 'saved\\file' + 'Node4.npy'
dict_load = np.load(dict_cdir4, allow_pickle=True).item()
Node4_FedBiLSTM_pre = dict_load['Node4_FedBiLSTM_pre']
Node4_FedBiLSTM_real = dict_load['Node4_FedBiLSTM_real']
dict_cdir44 = 'saved\\file' + 'Node4LSTM.npy'
dict_load44 = np.load(dict_cdir44, allow_pickle=True).item()
Node4_FedLSTM_pre = dict_load44['Node4_FedLSTM_pre']
dict_cdir444='saved\\fileNode4SingleLSTM.npy'
dict_load444=np.load(dict_cdir444, allow_pickle=True).item()
Node4_LSTM_pre = dict_load444['Node4_LSTM_pre']
dict_cdir4444='saved\\fileNode4SingleBiLSTM.npy'
dict_load4444=np.load(dict_cdir4444, allow_pickle=True).item()
Node4_BiLSTM_pre = dict_load4444['Node4_BiLSTM_pre']
plt.ioff()
plt.figure(1, figsize=(15, 2))
plt.rc('font', family='Times New Roman')
plt.clf()
# plt.grid(True, linestyle='--')
plt.plot(Node4_FedBiLSTM_real, label='Real Value', linewidth=1.5,marker=s_marker)  # color='#65A1D7'
plt.xlim(400, 1100)
plt.plot(Node4_LSTM_pre, label='LSTM', linewidth=1,linestyle=s_linestyle)
plt.plot(Node4_BiLSTM_pre, label='BiLSTM', linewidth=1,linestyle=s_linestyle)
plt.plot(Node4_FedLSTM_pre, label='Fed_LSTM', linewidth=1,linestyle=s_linestyle,color='#B9D4DB')  # color='#ED7D31'
plt.plot(Node4_FedBiLSTM_pre, label='Fed_BiLSTM', linewidth=1,linestyle=s_linestyle,color='#DE5B6D')
# plt.title("House 4", fontsize=20)
# plt.legend(loc='best')
# plt.xlabel('Time', fontsize=18)
plt.ylabel('House 4', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('Node4.png',dpi=500,bbox_inches='tight')
# plt.show()












dict_cdir5 = 'saved\\file' + 'Node5.npy'
dict_load = np.load(dict_cdir5, allow_pickle=True).item()
Node5_FedBiLSTM_pre = dict_load['Node5_FedBiLSTM_pre']
Node5_FedBiLSTM_real = dict_load['Node5_FedBiLSTM_real']
dict_cdir55 = 'saved\\file' + 'Node5LSTM.npy'
dict_load55 = np.load(dict_cdir55, allow_pickle=True).item()
Node5_FedLSTM_pre = dict_load55['Node5_FedLSTM_pre']
dict_cdir555='saved\\fileNode5SingleLSTM.npy'
dict_load555=np.load(dict_cdir555, allow_pickle=True).item()
Node5_LSTM_pre = dict_load555['Node5_LSTM_pre']
dict_cdir5555='saved\\fileNode5SingleBiLSTM.npy'
dict_load5555=np.load(dict_cdir5555, allow_pickle=True).item()
Node5_BiLSTM_pre = dict_load5555['Node5_BiLSTM_pre']
plt.ioff()
# plt.figure(1, figsize=(8, 5))
plt.figure(1, figsize=(15, 2))
plt.rc('font', family='Times New Roman')
plt.clf()
# plt.grid(True, linestyle='--')
plt.plot(Node5_FedBiLSTM_real, label='Real Value', linewidth=1.5,marker=s_marker)  # color='#65A1D7'
# plt.xlim(100, 200)
plt.xlim(400, 1100)
plt.plot(Node5_LSTM_pre, label='LSTM', linewidth=1,linestyle=s_linestyle)
plt.plot(Node5_BiLSTM_pre, label='BiLSTM', linewidth=1,linestyle=s_linestyle)
plt.plot(Node5_FedLSTM_pre, label='Fed_LSTM', linewidth=1,linestyle=s_linestyle,color='#B9D4DB')  # color='#ED7D31'
plt.plot(Node5_FedBiLSTM_pre, label='Fed_BiLSTM', linewidth=1,linestyle=s_linestyle,color='#DE5B6D')
# plt.title("House 5", fontsize=20)
# plt.legend(loc='best')
plt.xlabel('Time', fontsize=18)
# plt.ylabel('HEC', fontsize=18)
plt.ylabel('House 5', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('Node5.png',dpi=500,bbox_inches='tight')
# plt.show()


