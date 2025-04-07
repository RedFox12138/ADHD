import sys

from PlotFreq import PlotFreq
from PreProcess import preprocess, preprocess1
from SingleDenoise import remove_eog_with_visualization

#python库的路径
sys.path.append('D:\\anaconda\\lib\\site-packages')
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 定义频段
BANDS = ['delta', 'theta', 'alpha', 'beta']
BAND_RANGES = {
    'delta': (0.5, 4),
    'theta': (3, 11),
    'alpha': (8, 13),
    'beta': (10, 36)
}


def design_cheby2_bandpass(lowcut, highcut, fs, order=5, rs=40):
    """设计切比雪夫II型带通滤波器"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.cheby2(order, rs, [low, high], btype='band')
    return b, a


def compute_band_power(data, band, fs):
    """计算特定频段的功率"""
    lowcut, highcut = BAND_RANGES[band]
    b, a = design_cheby2_bandpass(lowcut, highcut, fs)
    # 应用滤波器
    filtered_data = signal.filtfilt(b, a, data)
    # if(band=="beta"):
    #     PlotFreq(data, 250)
    #     PlotFreq(filtered_data,250)
    #     plt.show()
    # 计算功率 (μV²)
    power = np.mean(filtered_data **  2)
    return power


def compute_band_powers(data):
    """计算所有频段的功率"""
    powers = {}
    for band in BANDS:
        powers[band] = compute_band_power(data, band, fs)

    # 计算theta/beta比值
    powers['theta_beta_ratio'] = powers['theta'] / powers['beta'] if powers['beta'] > 0 else 0

    return powers


# 加载数据
eeg = np.loadtxt('D:\\Pycharm_Projects\\ADHD\\data\\oksQL7aHWZ0qkXkFP-oC05eZugE8\\0406Game04凝胶.txt')
fs = 250  # 采样率
window_size = fs * 2  # 2秒窗口（500个点）

# 分窗处理
num_windows = len(eeg) // window_size
band_power_history = {band: [] for band in BANDS}
band_power_history['theta_beta_ratio'] = []

for i in range(num_windows):
    seg = eeg[i * window_size: (i + 1) * window_size]
    # plt.figure()
    seg= preprocess(seg,250)[20:-20]
    # plt.plot(seg)
    # seg,_ = remove_eog_with_visualization(seg,250,0)
    # plt.plot(seg)
    # plt.show()

    powers = compute_band_powers(seg)

    # 存储结果
    for band in BANDS:
        band_power_history[band].append(powers[band])
    band_power_history['theta_beta_ratio'].append(powers['theta_beta_ratio'])

# 可视化结果（5个子图垂直排列）
plt.figure(figsize=(12, 12))
colors = {'delta': 'blue', 'theta': 'green', 'alpha': 'orange', 'beta': 'red'}

# 绘制各频段功率
for i, band in enumerate(BANDS):
    plt.subplot(5, 1, i + 1)
    plt.plot(band_power_history[band], '.-', color=colors[band])
    plt.ylabel(f'{band} power (μV²)')
    plt.title(f'{band} band ({BAND_RANGES[band][0]}-{BAND_RANGES[band][1]}Hz) power')
    plt.grid(True)

# 添加θ/β比值到最后一个子图
# 添加θ/β比值到最后一个子图
plt.figure()
x = np.arange(len(band_power_history['theta_beta_ratio']))
y = band_power_history['theta_beta_ratio']

# 绘制原始数据点
plt.plot(x, y, 'm.-', label='θ/β ratio')

# 计算并绘制拟合线（使用3阶多项式拟合）
if len(x) > 3:  # 确保有足够的数据点进行拟合
    coeffs = np.polyfit(x, y, 3)
    poly = np.poly1d(coeffs)
    y_fit = poly(x)
    plt.plot(x, y_fit, 'b-', linewidth=2, label='Thrend')


plt.ylabel('θ/β ratio')
plt.xlabel('Windows (time/2s)')
plt.title('Theta/Beta ratio')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()