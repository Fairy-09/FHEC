from _DATA import Data
from _FSOP import FSOP


if __name__ == '__main__':
	
	startNum = 0
	trainNum = 12 * 700
	testNum = 12 * 100
	aheadNum = 8
	
	if input('Do you want to load & process data? [yes\\no] \n'
	         'If you already have done it, please input \'no\'\n') == 'yes':
		print('The full dataset shall be loaded and processed immediately.')
		# for i in range(5):
		# 	Data(i, 1, startNum, trainNum, testNum, aheadNum)
		# 	# Data(i, 2, startNum, trainNum, testNum, aheadNum)
		# 	# Data(i, 4, startNum, trainNum, testNum, aheadNum)
		# 	# Data(i, 6, startNum, trainNum, testNum, aheadNum)
		# 	# Data(i, 12, startNum, trainNum, testNum, aheadNum)
		for i in range(1):
			Data(i, 1, startNum, trainNum, testNum, aheadNum)
			# Data(i, 2, startNum, trainNum, testNum, aheadNum)
			# Data(i, 4, startNum, trainNum, testNum, aheadNum)
			# Data(i, 6, startNum, trainNum, testNum, aheadNum)
			# Data(i, 12, startNum, trainNum, testNum, aheadNum)
	else:
		print('The data will not be loaded.')
	
	# for i in range(5):
	# 	FSOP(8, i, 1)
	# 	# FSOP(8, i, 2)
	# 	# FSOP(8, i, 4)
	# 	# FSOP(8, i, 6)
	# 	# FSOP(8, i, 12)
	for i in range(1):
		FSOP(8, i, 1)
		# FSOP(8, i, 2)
		# FSOP(8, i, 4)
		# FSOP(8, i, 6)
		# FSOP(8, i, 12)