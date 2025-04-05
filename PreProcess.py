import sys #python库的路径

from PyEMD import EEMD, CEEMDAN
from numpy import std
from scipy.stats import stats, kurtosis
from sklearn.decomposition import FastICA

from Entropy import SampleEntropy2

sys.path.append('D:\\anaconda\\lib\\site-packages')
from matplotlib import pyplot as plt
from scipy import signal
import numpy as np
from scipy.signal import cheby2, filtfilt, welch

# 使用非交互式后端
plt.switch_backend('TkAgg')

# 预处理函数
def preprocess1(OriginalSignal, fs=250):
    # out = input.shape[0]
    # step1: 滤波
    d1 = OriginalSignal
    b, a = signal.butter(6, 0.5 / (fs / 2), 'highpass')  # 0.5Hz 高通巴特沃斯滤波器
    d1 = signal.filtfilt(b, a, d1)

    b, a = signal.butter(6, [49 / (fs / 2), 51 / (fs / 2)], 'bandstop')  # 50Hz 工频干扰
    d1 = signal.filtfilt(b, a, d1)

    b, a = signal.butter(6, [99 / (fs / 2), 101 / (fs / 2)], 'bandstop')  # 50Hz 工频干扰
    d1 = signal.filtfilt(b, a, d1)

    b, a = signal.butter(6, 40/ (fs / 2), 'lowpass')  # 100Hz 低通
    d1 = signal.filtfilt(b, a, d1)

    theta_band = [4,8]
    beta_band = [13,30]
    power_ratio = compute_power_ratio(d1, fs, theta_band, beta_band)

    return d1


def preprocess(raw_signal, fs=250, visualize=False):
    """完整的眼电信号预处理流程"""
    # ===== 1. 智能延拓 =====
    max_filter_len = 3 * 71  # 取FIR滤波器长度的3倍
    pad_len = int(1.5 * max_filter_len)

    # 镜像延拓 + 汉宁窗过渡
    padded = np.pad(raw_signal, (pad_len, pad_len), mode='reflect')
    window = np.concatenate([
        np.hanning(2 * pad_len)[:pad_len],
        np.ones(len(padded) - 2 * pad_len),
        np.hanning(2 * pad_len)[pad_len:]
    ])
    padded *= window

    # ===== 2. 滤波器设计 =====
    # 0.5Hz高通 (Butterworth)
    sos_high = signal.butter(4, 0.5, 'highpass', fs=fs, output='sos')

    # 50Hz带阻 (Notch)
    def design_notch(f0, Q=30):
        nyq = fs / 2
        w0 = f0 / nyq
        b, a = signal.iirnotch(w0, Q)
        return b, a

    # 40Hz低通 (FIR)
    fir_low = signal.firls(71, [0, 38, 42, fs / 2], [1, 1, 0, 0], fs=fs)

    # ===== 3. 零相位滤波链 =====
    filtered = signal.sosfiltfilt(sos_high, padded)  # 高通
    for f0 in [50, 100]:  # 消除基波和谐波
        b, a = design_notch(f0)
        filtered = signal.filtfilt(b, a, filtered)
    filtered = signal.filtfilt(fir_low, [1.0], filtered)  # 低通

    # ===== 4. 精准截断 =====
    result = filtered[pad_len:-pad_len]

    return result

#本来的max为10
def perform_ceemdan(signal, max_imf=15):
    ceemdan = CEEMDAN(max_imf=max_imf, trials=50, noise_strength=0.02)
    imfs = ceemdan(signal)
    residue = signal - np.sum(imfs, axis=0)
    return imfs[:max_imf], residue  # 二次保险


def perform_ceemdan_ica_denoising(signal,imfs, residue, ica_components=4,
                                  kurt_threshold=None, plot_results=True):


    # 2. 对IMF进行ICA处理
    ica = FastICA(n_components=min(ica_components, len(imfs)),
                  random_state=42, whiten='unit-variance')
    ica_sources = ica.fit_transform(imfs.T)  # shape: (n_samples, n_components)

    # 3. 眼电成分识别 (基于峰度和平滑度)
    kurts = []
    component_features = []
    for i in range(ica_sources.shape[1]):
        component = ica_sources[:, i]

        # 计算特征指标
        kurt = stats.kurtosis(component)
        diff_ratio = np.std(np.diff(component)) / np.std(component)
        score = 0.6 * abs(kurt) + 0.4 * diff_ratio

        kurts.append(kurt)
        component_features.append({
            'kurtosis': kurt,
            'diff_ratio': diff_ratio,
            'score': score
        })

    # 4. 确定要保留的成分
    kurts = np.array(kurts)
    # keep_indices = np.where((kurts <= kurt_threshold[0]) | (kurts >= kurt_threshold[1]))[0]  # 关键修改：| 代替 &
    keep_indices = np.where(kurts <= kurt_threshold[0])[0]
    removed_indices = np.setdiff1d(np.arange(ica_sources.shape[1]), keep_indices)

    # 5. 信号重建
    clean_ica_sources = np.zeros_like(ica_sources)
    clean_ica_sources[:, keep_indices] = ica_sources[:, keep_indices]  # 只保留极端峰度成分

    # 执行逆变换
    reconstructed_imfs = ica.inverse_transform(clean_ica_sources).T
    clean_signal = np.sum(reconstructed_imfs, axis=0) + residue

    # 6. 可视化
    if plot_results:
        plt.figure(figsize=(15, 14))

        # 原始信号
        plt.subplot(4, 1, 1)
        plt.plot(signal, color='darkblue')
        plt.title("Original Signal", fontsize=12, pad=10)
        plt.grid(alpha=0.3)

        # IMFs
        plt.subplot(4, 1, 2)
        for i, imf in enumerate(imfs):
            offset = i * 0.8
            plt.plot(imf + offset, label=f'IMF {i + 1}', linewidth=1)
            plt.text(len(imf) + 50, offset,
                     f"Range: {np.max(imf) - np.min(imf):.2f}",
                     va='center', fontsize=9)
        plt.yticks([])
        plt.title("CEEMDAN IMF Components", fontsize=12, pad=10)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.grid(alpha=0.3)

        # ICA成分及分数标注
        plt.subplot(4, 1, 3)
        max_amp = np.max(np.abs(ica_sources))
        for i in range(ica_sources.shape[1]):
            offset = i * (max_amp * 4)
            color = 'red' if i in removed_indices else 'steelblue'
            plt.plot(ica_sources[:, i] + offset,
                     color=color, linewidth=1,
                     label=f'IC {i + 1} (Removed)' if i in removed_indices else f'IC {i + 1}')

            # 标注信息
            info = (f"Kurt: {kurts[i]:.2f}\n")
            plt.text(len(ica_sources) + 100, offset, info,
                     color=color, va='center', fontsize=9,
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

            plt.text(-50, offset, f'IC{i + 1}',
                     color=color, ha='right', va='center', fontsize=10,
                     bbox=dict(facecolor=color, alpha=0.2))

        plt.yticks([])
        plt.title(f"ICA Components (Red=Removed, Threshold={kurt_threshold})", fontsize=12, pad=10)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.grid(alpha=0.3)

        # 去噪结果对比
        plt.subplot(4, 1, 4)
        plt.plot(signal, color='darkblue', alpha=0.6, label='Original')
        plt.plot(clean_signal, color='green', linewidth=1.5, label='Denoised')

        if len(removed_indices) > 0:
            removed_text = f"Removed ICs: {removed_indices + 1}"
            plt.text(0.05, 0.9, removed_text,
                     transform=plt.gca().transAxes,
                     color='red', fontsize=10,
                     bbox=dict(facecolor='white', alpha=0.8))

        plt.title("Denoising Result Comparison", fontsize=12, pad=10)
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    return clean_signal, removed_indices.tolist()

def dynamic_range_ratio(signal, window_size=100):
    rolling_max = np.lib.stride_tricks.sliding_window_view(signal, window_size).max(axis=1)
    rolling_min = np.lib.stride_tricks.sliding_window_view(signal, window_size).min(axis=1)
    return np.median(rolling_max / rolling_min)  # 值越大，区分度越高

def kurtosis_variation(signal, segment_length=500):
    segments = np.array_split(signal, len(signal)//segment_length)
    kurt_values = [kurtosis(seg) for seg in segments]
    return np.std(kurt_values)/np.mean(kurt_values)  # 变异系数越大，局部差异越明显

def plot_imfs(signal, imfs, title, residue, min_threshold=0.2,threshold=0.4):
    """
    改进的可视化函数：每个IMF单独显示在一张图中，避免拥挤

    参数：
        signal : 原始信号
        imfs : IMF分量列表
        title : 图标题前缀
        residue : 残差项
        threshold : 保留IMF的阈值（默认0.9）
    """
    n_imfs = len(imfs)
    plt.figure(figsize=(14, 3 * (n_imfs + 2)))  # 动态调整总高度

    # 原始信号
    plt.subplot(n_imfs + 2, 1, 1)
    plt.plot(signal, color='blue')
    plt.title(f"{title} - Original Signal", pad=15)
    plt.grid(True, alpha=0.3)

    # 绘制各IMF（单独子图）
    reconstructed = np.zeros_like(residue)
    kept_indices = []

    for i, imf in enumerate(imfs):
        # 计算特征值（这里使用样本熵示例）
        # peak = SampleEntropy2(imf, 0.2 * np.std(imf))
        peak =  dynamic_range_ratio(imf) #阈值设成0就好
        # peak = kurtosis_variation(imf) #阈值设成0就好

        color = 'green' if abs(peak) <= 0.8 else 'red'


        # 创建独立子图
        ax = plt.subplot(n_imfs + 2, 1, i + 2)
        plt.plot(imf, color=color, linewidth=1)

        # 添加标注（自动避开信号峰值区域）
        y_pos = 0.8 * np.max(imf) if np.max(imf) > 0 else 0.8 * np.min(imf)
        x_pos = len(imf) // 5  # 20%位置

        plt.text(x_pos, y_pos,
                 f"IMF {i + 1} | Diff Value: {peak:.3f}",
                 color='black',
                 bbox=dict(facecolor=color, alpha=0.2, edgecolor='none'))

        plt.title(f"IMF {i + 1} ({'KEPT' if (peak >= threshold or peak <=min_threshold)else 'DISCARDED'})",
                  pad=10, color=color)
        plt.grid(True, alpha=0.3)

        # if peak >= threshold or peak<=min_threshold:
        if abs(peak) <= 0.8:
            reconstructed += imf
            kept_indices.append(i)

    # 残差项
    plt.subplot(n_imfs + 2, 1, n_imfs + 2)
    plt.plot(residue, color='purple')
    plt.title("Residue Component", pad=10)
    plt.grid(True, alpha=0.3)

    plt.tight_layout(h_pad=3.0)  # 增加子图间距
    plt.show()

    return reconstructed, kept_indices

def calculate_autocorr_peak(imf):
    """计算IMF的自相关最大值（排除零滞后点）"""
    autocorr = signal.correlate(imf, imf, mode='same')
    autocorr = autocorr / np.max(autocorr)  # 归一化到[0,1]
    center = len(autocorr) // 2
    return np.max(autocorr[center+1:])  # 排除零滞后点


def compute_power_ratio(eeg_data, Fs, theta_band, beta_band):
    """
    计算 theta 波段和 beta 波段的功率比，并绘制频谱对比图

    参数:
    eeg_data: 输入的 EEG 信号（一维数组）
    Fs: 采样频率
    theta_band: theta 波段范围 [f1_theta, f2_theta]
    beta_band: beta 波段范围 [f1_beta, f2_beta]

    返回:
    power_ratio: theta 和 beta 波段的功率比
    """

    # 切比雪夫 II 型滤波器参数
    Rs = 20  # 阻带衰减（dB）

    # 设计 theta 波段的切比雪夫 II 型带通滤波器
    f1_theta = theta_band[0] / (Fs / 2)
    f2_theta = theta_band[1] / (Fs / 2)
    Wn_theta = [f1_theta, f2_theta]
    b_theta, a_theta = cheby2(6, Rs, Wn_theta, btype='bandpass')  # 8 是滤波器阶数

    # 设计 beta 波段的切比雪夫 II 型带通滤波器
    f1_beta = beta_band[0] / (Fs / 2)
    f2_beta = beta_band[1] / (Fs / 2)
    Wn_beta = [f1_beta, f2_beta]
    b_beta, a_beta = cheby2(6, Rs, Wn_beta, btype='bandpass')  # 8 是滤波器阶数

    # 对 theta 波段滤波
    theta_filtered = filtfilt(b_theta, a_theta, eeg_data.astype(float))
    # 对 beta 波段滤波
    beta_filtered = filtfilt(b_beta, a_beta, eeg_data.astype(float))

    # 计算 theta 和 beta 波段功率
    theta_power = np.sum(theta_filtered ** 2)
    beta_power = np.sum(beta_filtered ** 2)

    # 计算 theta 和 beta 功率比
    if beta_power != 0:
        power_ratio = theta_power / beta_power
    else:
        power_ratio = np.nan  # 防止除以零

    # # 绘制频谱对比图
    # plot_spectrum_comparison(eeg_data, theta_filtered, beta_filtered, Fs, theta_band, beta_band)

    return power_ratio

