import sys

import pywt
from sympy.abc import alpha

from PlotFreq import PlotFreq
from Wavelet import wavelet_decomposition, compute_frequency_ranges, ewt_decomposition

#python库的路径
sys.path.append('D:\\anaconda\\lib\\site-packages')
import matplotlib.pyplot as plt
import numpy as np
from PreProcess import preprocess, perform_ceemdan, plot_imfs, perform_ceemdan_ica_denoising, preprocess1
from scipy.signal import cheby2, filtfilt, welch
# data = []
# with open('D:\\pycharm Project\\WX\\data\\oksQL7aHWZ0qkXkFP-oC05eZugE8\\raw_20250324_094828_076.txt', 'r', encoding='utf-8') as file:
#     for line in file:
#         values = line.strip().split()  # 去掉换行符并按空格分割
#         data.append(float(values))  # 打印分割后的列表

# data1 = np.loadtxt('data/oksQL7aHWZ0qkXkFP-oC05eZugE8/0327橙色眨眼1.txt', dtype=np.float32)
# data2 = np.loadtxt('D:\\pycharm Project\\WX\\data\\oksQL7aHWZ0qkXkFP-oC05eZugE8\\干电极2.txt',dtype=np.float32)
# data3 = np.loadtxt('D:\\pycharm Project\\WX\\data\\oksQL7aHWZ0qkXkFP-oC05eZugE8\\凝胶1.txt',dtype=np.float32)
# data4 = np.loadtxt('D:\\pycharm Project\\WX\\data\\oksQL7aHWZ0qkXkFP-oC05eZugE8\\凝胶2.txt',dtype=np.float32)
#
# data1 = np.array(data1).flatten()
# data2 = np.array(data2).flatten()
# data3 = np.array(data3).flatten()
# data4 = np.array(data4).flatten()


# plt.figure()
# plt.subplot(4, 1, 1)
# plt.plot(data1)
# plt.subplot(4, 1, 2)
# plt.plot(data2)
# plt.subplot(4, 1, 3)
# plt.plot(data3)
# plt.subplot(4, 1, 4)
# plt.plot(data4)
# plt.show()

fs=250
data = np.loadtxt('凝胶2.txt', dtype=np.float32)



window_size = 2* fs  # 2秒窗长

# 加载数据a1
eeg = np.loadtxt('凝胶2.txt')

# 分窗处理
num_windows = len(eeg) // window_size
all_cleaned = []
all_eog = []

plt.figure(figsize=(12, 8))
for i in range(num_windows):
    start = i * window_size
    end = start + window_size
    seg = eeg[start:end]
    seg = preprocess(seg)
    cleaned = seg
    all_cleaned.extend(cleaned)
    # 绘制每个窗口
    plt.subplot(num_windows, 1, i + 1)
    plt.plot(seg, 'b', label='Original', alpha=0.5)
    plt.plot(cleaned, 'r', label='Cleaned')
    plt.legend()



processed_points = preprocess(data, fs)
plt.figure()
# plt.subplot(2,1,1)
# plt.plot(all_cleaned,label='Processed by Windows', color='red')
# plt.plot(processed_points,label='Processed by All', color='blue')

# plt.subplot(2,1,2)
ax = PlotFreq(data, fs=fs, label='Raw', color='red')
# # 绘制预处理信号频谱（红色）
PlotFreq(processed_points, fs=fs, label='Processed', color='blue', ax=ax)
plt.legend(loc='upper right')
# plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()










