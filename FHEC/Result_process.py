import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import csv
#类的括号里表示该类是从哪个类继承下来的。object类是所有类都会继承的类。
#通过类名加()表示创建类的对象---实例。eg:student=Student()
class resultProcess():
	def __init__(self, order, interval, step_num, if_show):
		#有__init__方法时，类实例化需要传入和__init__方法匹配的参数
		#self是对对象自身的引用。加了self是实例变量，作用域为当前实例。不加self是局部变量，作用域为当前函数。
		#在外部将resultProcess类进行实例化时，可以调用加了self的，不可以调用不加self的，因为不加self的是局部变量，只能在当前函数(此处为__init__)内调用
		self.order = order
		#将外部传来的参数order的值给resultProcess类自己的属性变量self.order
		self.interval = interval
		self.step_num = step_num
		self.if_show = if_show
		
		self.dict_cdir = 'saved\\predict_house' + str(self.order)\
		                 + '_interval' + str(self.interval * 5) + 'min_ts'\
		                 + str(self.step_num) + '.npy'
		self.dict_load = np.load(self.dict_cdir, allow_pickle=True).item()
		#np.load()读取具有npy扩展名磁盘文件，加.item（）返回字典类型，后不加.item()是一个ndarray类型
		#allow_pickle=True的作用是从磁盘文件读出来之前对对象进行序列化(变成二进制即byte数组)或反序列化(此处应是反序列化的作用)
		
		self.predict_list = self.dict_load['result_list']
		self.result_print = self.dict_load['result_print']
		self.result_save = self.dict_load['result_save']
		
		self.y_real = self.dict_load['y_real']
		
		self.name = self.dict_load['name']
		self.print_name = self.dict_load['print']
		#turquoise
		self.colors = ['aqua',
		               'salmon',
		               'grey',
		               'orange',
		               'royalblue',
		               'gold',
		               'violet',
		               '#855a8a',
		               'limegreen',
		               'purple',
		               'seagreen',
		               'yellow',
		               'lightcoral',
		               'indianred',
		               'steelblue',
		               'tomato',
		               'slategrey',
		               'red',
		               'azure',
		               'tan',
		               'skyblue',
		               'chartreuse',
		               'lavender',
		               'papayawhip',
		               'gainsboro',
		               'navajowhite',
		               'thistle',
		               'teal',
		               'indigo',
		               'cornsilk',
		               'sienna',
		               'dodgerblue'
		               'sage',
		               'darkmagenta']


		self.want_list = ['dTr',
		                  'rDf',
		                  'MLP',
		                  'SVR',
		                  'LSTM',
		                  'BiLSTM'
		                  ]

		self.predict_want = []
		self.result_want = []
		
		self.metrics_list = None
		self.mae_list = None
		self.rmse_list = None
		self.mape_list = None
		self.r2_list = None
		
		self.name_full_list = None
		self.name_short_list = None
		self.name_print_list = None
	
	###############################################################################
	
	def __select_want(self):
		for i in range(len(self.predict_list)):
			name = self.predict_list[i][0][1]
			if name in self.want_list:
				self.predict_want.append(self.predict_list[i])
				self.result_want.append(self.result_save[i])
	
	def __check_name(self):
		for i in range(len(self.predict_want)):
			name = list(self.predict_want[i][0][0])
			break_mark = []
			for j in range(len(name) - 1):
				if i < 2:
					if name[j + 1].isupper():
						break_mark.append(j + 1)
				else:
					if name[j + 1] == '-':
						break_mark.append(j + 2)
			
			for k in range(len(break_mark)):
				name.insert(k + break_mark[k], '\n')
			
			self.predict_want[i][0].append(''.join(name))
			
	def __result_split(self):
		result_all = []
		for k in range(len(self.result_want)):
			result_all.append(self.result_want[k][:5])
		result_all = np.array(result_all).T.tolist()
		
		self.metrics_list = np.array(result_all[1:]).astype(float).tolist()
		self.mae_list = self.metrics_list[0]
		self.rmse_list = self.metrics_list[1]
		self.mape_list = self.metrics_list[2]
		self.r2_list = self.metrics_list[3]
	
	def __name_split(self):
		self.name_full_list = []
		self.name_short_list = []
		self.name_print_list = []
		for i in range(len(self.predict_want)):
			self.name_full_list.append(self.predict_want[i][0][0])
			self.name_short_list.append(self.predict_want[i][0][1])
			self.name_print_list.append(self.predict_want[i][0][2])
	
	def PREPARE(self):
		self.__select_want()
		self.__check_name()
		self.__result_split()
		self.__name_split()
			
	###############################################################################
	
	def __abs_sub(self, data_A, data_B):
		data_A = data_A.tolist()
		data_B = data_B.tolist()
		
		result = []
		for i in range(len(data_A)):
			result.append(abs(data_A[i] - data_B[i]))
		return np.array(result)
	
	def DRAW_BOX(self):

		error_list = []
		for i in range(len(self.predict_want)):
			error_list.append(self.__abs_sub(self.y_real, self.predict_want[i][1]))
		colors = self.colors[:len(self.predict_want)]

		plt.figure(figsize=(16, 4))  # 设置画布的尺寸
		plt.rc('font', family='Times New Roman')
		plt.rcParams["font.weight"] = "bold"
		plt.ylabel("Absolute Error", fontsize=18)
		box_list = plt.boxplot(error_list,
		                       labels=self.name_print_list,
		                       notch=True,
		                       showfliers=False,
		                       patch_artist=True,
		                       boxprops={'color': 'black', 'facecolor': 'pink'})

		for patch, color in zip(box_list['boxes'], colors):
			patch.set_facecolor(color)
		for box in box_list['boxes']:
			box.set(linewidth=3)
		for whisker in box_list['whiskers']:
			whisker.set(linewidth=2)
		for cap in box_list['caps']:
			cap.set(linewidth=2)
		for median in box_list['medians']:
			median.set(linewidth=2)
		plt.subplots_adjust(left=0.05, bottom=0.095, right=0.99, top=0.99)
		plt.grid(True, linestyle=":", color="gray", linewidth="0.5", axis='y')

		if self.if_show == 1:
			plt.savefig('result\\Draw_box_' +
			            str(self.order) + '_' +
			            str(self.interval) + '_' +
			            str(self.step_num) + '.png')
			plt.close(1)
		else:
			plt.show()

	###############################################################################
	
	def __linear_coefs(self, dataX, dataY):
		points = np.c_[dataX, dataY]
		#np.c_是按行连接两个矩阵，把两个矩阵左右相加，要求行数相等，类似于pandas中的merge()
		M = len(points)    #points的行数
		x_bar = np.mean(points[:, 0])   #取第一列，算其平均值
		sum_yx = 0
		sum_x2 = 0
		sum_delta = 0
		for i in range(M):
			x = points[i, 0]
			y = points[i, 1]
			sum_yx += y * (x - x_bar)      #不用管，做的是线性回归的事
			sum_x2 += x ** 2
		w = sum_yx / (sum_x2 - M * (x_bar ** 2))

		for i in range(M):
			x = points[i, 0]
			y = points[i, 1]
			sum_delta += (y - w * x)      #不用管，做的是线性回归的事
		b = sum_delta / M
		return w, b

	def __scatter_and_linear(self, real, predict, name, index):
		w, b = self.__linear_coefs(real, predict)
		margin = abs(min(real) - max(real)) * 0.025     #为了把y=x这条线撑开来，使图好看

		plt.style.use('seaborn-darkgrid')
		plt.figure(1, figsize=(8, 4))  # wide

		plt.rc('font', family='Times New Roman')
		plt.rcParams["font.weight"] = "bold"

		x = np.linspace(min(real) - margin, max(real) + margin, 400)  #y=x由点阵组成
		y = x
		# plt.plot(x, y, color='gray', linewidth=2, linestyle=":")
		# plt.plot(x, y, color='#B57EC8', linewidth=3, linestyle="-.")
		plt.plot(x, y, color='#8888C6', linewidth=2, linestyle="-.")

		# plt.scatter(real, predict, s=18, color='white', edgecolors='red')
		plt.scatter(real, predict, s=80, color='#CA7DB4', edgecolors='#CA7DB4',alpha=0.5,marker='*')

		x = np.linspace(min(real), max(real), len(real))
		y = x * w + b
		# plt.plot(x, y, color='purple', linewidth=3, linestyle="--")
		plt.plot(x, y, color='#C65256', linewidth=2, linestyle="--")

		# plt.grid(True, linestyle=":", color="lightgray", linewidth=1, axis='both')
		plt.grid(True, linestyle="-", color="white", linewidth=1, axis='both')

		plt.xlabel("Real Data", fontsize=20)
		plt.ylabel("Prediction", fontsize=20)
		plt.xlim(min(real) - margin, max(real) + margin)
		plt.ylim(min(real) - margin, max(real) + margin)

		plt.subplots_adjust(left=0.09, bottom=0.13, right=0.995, top=0.89)
		plt.title(name + ": Y=" + str(round(w, 2)) + "X+" + str(round(b, 2)), fontsize=32)

		if self.if_show == 1:
			plt.savefig('result\\Draw_scatter_' +
			            str(self.order) + '_' +
			            str(self.interval) + '_' +
			            str(self.step_num) + '_' +
			            str(index) + '.png')
			plt.close(1)
		else:
			plt.show()

	def DRAW_SCATTER(self):
		for i in range(len(self.predict_want)):
			self.__scatter_and_linear(self.y_real, self.predict_want[i][1], self.name_full_list[i], str(i + 1))
	
	###############################################################################
	
	def __bar_model(self):

		result_list = []

		for i in range(len(self.metrics_list)):
			result_list.append(np.array(self.metrics_list[i]).astype(float))
			scaler = MinMaxScaler(feature_range=(1, 2))
			result_list[i] = scaler.fit_transform(result_list[i].reshape(-1, 1))
			result_list[i] = result_list[i].reshape(-1, ).tolist()
		result_list = np.array(result_list).T.tolist()

		# 输入统计数据
		metrices = ['MAE', 'RMSE', 'MAPE', 'R$^2$']

		colors = self.colors[:len(self.want_list)]
		labels = self.want_list

		plt.figure(1, figsize=(16, 6))

		plt.rc('font', family='Times New Roman')
		plt.rcParams["font.weight"] = "bold"

		bar_width = 0.2  # 条形宽度
		index_list = np.linspace(1, 10, 4)
		rect = plt.bar(x=index_list, height=result_list[0], width=bar_width,
		               color=colors[0], label=labels[0], edgecolor='dimgray', align='edge', linewidth=1.5)
		# autolabel(rect)
		for l1 in range(len(result_list) - 2):
			index_list = index_list + bar_width
			rect = plt.bar(x=index_list, height=result_list[l1 + 1], width=bar_width,
			               color=colors[l1 + 1], label=labels[l1 + 1], edgecolor='dimgray', align='edge', linewidth=1.5)
		# autolabel(rect)
		index_list = index_list + bar_width
		rect = plt.bar(x=index_list, height=result_list[-1], width=bar_width,
		               color=colors[-1], label=labels[-1], edgecolor='black', align='edge', linewidth=2)

		plt.legend(fontsize=12)
		plt.xticks(index_list - bar_width * (len(result_list) - 1) * 0.5, metrices,
		           fontsize=18)
		plt.xlim((0.50, 14.75))
		plt.title('Relative comparison of forecasting performance.', fontsize=18)
		plt.grid(True, linestyle=":", color="gray", linewidth="0.5", axis='y')
		plt.subplots_adjust(left=0.03, bottom=0.06, right=0.98, top=0.93)
		
		if if_show == 1:
			plt.savefig('result\\Draw_bar_' +
			            str(self.order) + '_' +
			            str(self.interval) + '_' +
			            str(self.step_num) + '.png')
			plt.close(1)
		else:
			plt.show()

	def DRAW_BAR(self):
		self.__bar_model()

	###############################################################################
	
	def __find_limit(self, min, max):
		if min == 0:
			expand_min = 0
		elif min > 0:
			expand_min = abs(min) * 0.9 * (min / abs(min))
		else:
			expand_min = abs(min) * 1.1 * (min / abs(min))

		if max == 0:
			expand_max = 0
		elif max > 0:
			expand_max = abs(max) * 1.1 * (max / abs(max))
		else:
			expand_max = abs(max) * 0.9 * (max / abs(max))

		length = expand_max - expand_min

		i = 0
		if length > 1:
			while length > 10 ** (i + 1):
				i = i + 1
		# i = i - 1
		elif length < 1:
			while length < 10 ** (-(i + 1)):
				i = i + 1
			i = -(i + 1)

		limit_min = 10 ** i * ((expand_min // 10 ** i) - 1)
		if expand_min > 0 > limit_min:
			limit_min = 0
		limit_max = 10 ** i * ((expand_max // 10 ** i) + 1)
		limit_mid = (limit_max + limit_min) / 2

		return [limit_min, limit_mid, limit_max]

	def __projection_model(self, error_list, name, name_short):
		data_list = []
		data_list = data_list + error_list

		for i in range(len(data_list)):
			while data_list[i] < 0:
				data_list[i] = 0

		data_list.append(data_list[0])
		theta = np.arange(0, 2 * np.pi, 2 * np.pi / (len(data_list) - 1))
		theta = theta.tolist()
		theta.append(0)
		fig_limit = self.__find_limit(min(data_list), max(data_list))

		plt.figure(1, figsize=(6, 5))  # wide

		plt.rc('font', family='Times New Roman')

		ax = plt.subplot(projection='polar')
		ax.set_thetagrids(np.arange(0.0, 360.0, 360.0 / (len(data_list) - 1)),
		                  labels=self.name_print_list, weight="bold",
		                  color="black", fontsize=12)
		ax.set_rlabel_position(360 - 360.0 / (len(data_list) - 1))

		ax.set_rticks(fig_limit)
		ax.set_rlim(0, max([fig_limit[-1] * 1.1, (max(data_list)) * 1.2]))

		ax.set_theta_direction(-1)
		ax.set_theta_zero_location('N')
		ax.set_xticklabels(labels=self.name_print_list, y=-0.11, weight='light', fontsize=16)
		ax.plot(theta, data_list, '--', linewidth=2.5, marker='o')
		plt.title(name, fontsize=24, y=1.2)

		plt.subplots_adjust(left=0.0, bottom=0.1, right=1.0, top=0.77)  # wide

		if if_show == 1:
			plt.savefig('result\\Draw_projection_' +
			            str(self.order) + '_' +
			            str(self.interval) + '_' +
			            str(self.step_num) + '_' +
			            str(name_short) + '.png')
			plt.close(1)
		else:
			plt.show()

	def __rose_model(self, error_list, name, name_short):
		data_list = []
		data_list = data_list + error_list
		# data_list=(np.array(data_list)+abs(np.array(data_list)))/2
		# data_list=data_list.tolist()
		data_list.append(data_list[0])
		theta = np.linspace(0, np.pi * 2, len(data_list)).tolist()
		bar_width = 2 * np.pi / (len(data_list) - 1)
		fig_limit = self.__find_limit(min(data_list), max(data_list))

		colors = self.colors

		plt.figure(1, figsize=(6, 5),dpi=600)  # wide

		plt.rc('font', family='Times New Roman')

		ax = plt.subplot(projection='polar')
		ax.set_theta_direction(-1)
		ax.set_theta_zero_location('N')
		ax.set_thetagrids(np.arange(0.0, 360.0, 360.0 / (len(data_list) - 1)),
		                  labels=self.name_print_list, weight="bold",
		                  color="black", fontsize=12)
		# ax.set_rticks(fig_limit)
		ax.set_rticks([0.4,0.7,1])
		# ax.set_rlim(0, max([fig_limit[-1] * 1.1, (max(data_list)) * 1.2]))
		ax.set_rlim(0, 1.1)


		ax.set_rlabel_position(0)
		ax.set_xticklabels(labels=self.name_print_list, y=-0.11, weight='light', fontsize=16)
		for i in range(len(data_list)):
			ax.bar(theta[i], data_list[i], width=bar_width, color=colors[i % len(colors)], align='center', bottom=0,
			       edgecolor='dimgray', linewidth=2, alpha=0.85)

		# plt.title(name, fontsize=24, y=1.2)
		plt.title('House '+str(self.order+1), fontsize=24, y=1.2)

		plt.subplots_adjust(left=0.0, bottom=0.115, right=1.0, top=0.80)

		if if_show == 1:
			plt.savefig('result\\Draw_rose_' +
			            str(self.order) + '_' +
			            str(self.interval) + '_' +
			            str(self.step_num) + '_' +
			            str(name_short) + '.png')
			plt.close(1)
		else:
			plt.show()
			
	def DRAW_PROJECTION(self):
		self.__projection_model(self.mae_list, 'MAE', '0')
		# self.__projection_model(self.rmse_list, 'RMSE', '1')
		# self.__projection_model(self.mape_list, 'MAPE', '2')
		# self.__projection_model(self.r2_list, 'R2', '3')
		
	def DRAW_ROSE(self):
		# self.__rose_model(self.mae_list, 'MAE', '0')
		# self.__rose_model(self.rmse_list, 'RMSE', '1')
		# self.__rose_model(self.mape_list, 'MAPE', '2')
		self.__rose_model(self.r2_list, 'R2', '3')
	
	###############################################################################
	def CSV_PRINT(self):
		save_file_name = 'result\\Result_' + \
		                 str(self.order) + '_' + \
		                 str(self.interval) + '_' + \
		                 str(self.step_num) + '_' + '.csv'

		csv_file = open(save_file_name, "w", newline="")  # 创建csv文件
		writer = csv.writer(csv_file)  # 创建写的对象
		writer.writerow(["", "MAE", "RMSE", "MAPE", "R2", "ACC", "TIME"])
		for wtr in range(len(self.result_want)):
			writer.writerow(self.result_want[wtr])

	def EXCECUTE(self):
		
		self.PREPARE()
		
		# self.DRAW_BOX()
		# self.DRAW_SCATTER()
		# self.DRAW_PROJECTION()
		# self.DRAW_ROSE()
		# self.CSV_PRINT()
		print('Complete.')
		
	
class resultProcess_Plot(resultProcess):
	def __init__(self, order, interval, step_num, if_show, start_num, end_num):
		resultProcess.__init__(self, order, interval, step_num, if_show)
		self.start_num = start_num
		self.end_num = end_num
		
		self.main_linewidth = 2.5
		self.second_linewidth = 2
		self.third_linewidth = 1
	
	###############################################################################
	
	def __plot_predict(self):
		
		# plt.figure(1, figsize=(18, 4),dpi=600)
		plt.figure(1, figsize=(18, 4))
		plt.rc('font', family='Times New Roman',size=16)
		
		plt.plot(self.y_real[self.start_num:self.end_num],
		         'black',
		         label='Actual data',
		         linewidth=self.main_linewidth,
		         linestyle='--',
		         marker='o', markersize=5)

		for i in range(len(self.predict_want)):
			if i != len(self.predict_want)-1:
				plt.plot(self.predict_want[i][1][self.start_num:self.end_num],
				         self.colors[i],
				         label=self.name_full_list[i],
				         linewidth=self.third_linewidth)
			else:
				plt.plot(self.predict_want[i][1][self.start_num:self.end_num],
				         self.colors[i],
				         label=self.name_full_list[i],
				         linewidth=self.second_linewidth,
				         marker='s', markersize=3)

		# plt.xlim((-(6//self.interval), self.end_num+(6//self.interval)))
		plt.xlim((-(2 // self.interval), self.end_num-self.start_num + (2// self.interval)))

		plt.ylim(0, )
		plt.subplots_adjust(left=0.05, bottom=0.125, right=0.99, top=0.99)
		plt.xlabel("Time ("+str(self.interval*5)+" minutes)", fontsize=18)
		plt.ylabel("Energy Consumption (kWh)", fontsize=18)
		# plt.title('Performance Comparison of Household '+str(order))
		# plt.xticks(np.arange(0, self.end_num + 1, 12//self.interval))
		plt.xticks(np.arange(0, self.end_num-self.start_num + 1, 12 // self.interval),fontsize=14)
		plt.grid(True, linestyle=":", color="gray", linewidth="0.5", axis='both')
		# plt.legend(loc='center right', ncol=1, fontsize=12)
		# plt.legend(loc='best', ncol=2, fontsize=12)

		if self.if_show == 1:
			plt.savefig('result\\Draw_plot_' +
			            str(self.order) + '_' +
			            str(self.interval) + '_' +
			            str(self.step_num) + '.png',dpi=600)
			plt.close(1)
		else:
			plt.show()
			
	def __plot_legend(self):
		
		plt.figure(1, figsize=(15, 2))

		plt.rc('font', family='Times New Roman')

		plt.plot(self.y_real,
		         'black',
		         label='Actual data',
		         linewidth=self.main_linewidth*3,
		         linestyle='--',
		         marker='o', markersize=5*3)

		for i in range(len(self.predict_want)):
			if i != len(self.predict_want)-1:
				plt.plot(self.predict_want[i][1],
				         self.colors[i],
				         label=self.name_full_list[i],
				         linewidth=self.third_linewidth*3)
			else:
				plt.plot(self.predict_want[i][1],
				         self.colors[i],
				         label=self.name_full_list[i],
				         linewidth=self.second_linewidth*3,
				         marker='s', markersize=3*3)

		plt.xlim(-1000, -1001)
		plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
		plt.legend(loc='center right', ncol=5, fontsize=18)
		
		if self.if_show == 1:
			plt.savefig('result\\Draw_plot_legend.png',dpi=600)
			plt.close(1)
		else:
			plt.show()
		
	def DRAW_PLOT(self):
		self.__plot_predict()
		self.__plot_legend()

	###############################################################################

	def EXCECUTE_plot(self):
		self.PREPARE()
		self.DRAW_PLOT()
		

def Result_print(order, interval, step_num, if_show):
	resultProcess(order, interval, step_num, if_show).EXCECUTE()


def Result_print_plot(order, interval, step_num, if_show, start_num, end_num):
	resultProcess_Plot(order, interval, step_num, if_show, start_num, end_num).EXCECUTE_plot()

#__name__='__main__'作用是解决当前脚本是否被直接调用的问题。当被python直接调用时执行，被别人引入时不执行
if __name__ == '__main__':
	
	if_show = 1
	#plot_range的shape是(5,5,2)  x,y,z表示x个y行z列的矩阵
	plot_range = [[[12, 133], [48, 144], [24, 72], [12, 48], [0, 24]],
	              [[96, 288], [48, 144], [24, 72], [12, 48], [0, 24]],
	              [[670, 288], [330, 144], [160, 72], [106, 48], [47, 24]],
	              [[96, 288], [48, 144], [24, 72], [12, 48], [0, 24]],
	              [[312, 288], [156, 144], [78, 72], [48, 48], [18, 24]]]
	
	# Result_print(0, 1, 8, if_show)
	Result_print_plot(0, 1, 8, if_show, plot_range[0][0][0], plot_range[0][0][1])
	
	for i in range(5):
		Result_print(i, 1, 8, if_show)
		# Result_print(i, 2, 8, if_show)
		# Result_print(i, 4, 8, if_show)
		# Result_print(i, 6, 8, if_show)
		# Result_print(i, 12, 8, if_show)

	for i in range(5):
		Result_print_plot(i, 1, 8, if_show, plot_range[i][0][0], plot_range[i][0][1])
		# Result_print_plot(i, 2, 8, if_show, plot_range[i][1][0], plot_range[i][1][1])
		# Result_print_plot(i, 4, 8, if_show, plot_range[i][2][0], plot_range[i][2][1])
		# Result_print_plot(i, 6, 8, if_show, plot_range[i][3][0], plot_range[i][3][1])
		# Result_print_plot(i, 12, 8, if_show, plot_range[i][4][0], plot_range[i][4][1])

	#extra knowledge
	#若实例的变量名以__开头表明是一个私有变量，只有内部可以访问，外部不可以访问
	#这样使外部代码不可以随意改变对象内部的状态
	#若外部代码要获取私有变量__score，可以在类中加入get_score方法
	#def get_score(self):
	#   return self.__score
	#同理如果要修改，则加入set_score方法
	#def set_score(self,score):
	#   self.__score=score