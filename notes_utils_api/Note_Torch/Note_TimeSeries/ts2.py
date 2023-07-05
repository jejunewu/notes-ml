import numpy as np
import tushare as ts

data_close = ts.get_k_data('000001', start='2018-01-01', index=True)['close'].values  # 获取上证指数从20180101开始的收盘价的np.ndarray
data_close = data_close.astype('float32')  # 转换数据类型
# 将价格标准化到0~1
max_value = np.max(data_close)
min_value = np.min(data_close)
data_close = (data_close - min_value) / (max_value - min_value)

print(data_close)