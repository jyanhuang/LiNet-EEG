import torch
import numpy as np
from scipy.signal import butter, filtfilt

# =======Bands extraction (bandpass filter)======
def bandpass_filter_np(x, fs, low, high, order=4):
    """
    支持 x 形状 (T,) 或 (C,T)
    """
    nyq = fs / 2.0
    low_norm = low / nyq
    high_norm = high / nyq
    b, a = butter(order, [low_norm, high_norm], btype="band")

    x = np.asarray(x)
    if x.ndim == 1:
        return filtfilt(b, a, x)
    elif x.ndim == 2:
        return filtfilt(b, a, x, axis=-1)
    else:
        raise ValueError(f"bandpass_filter_np expects (T,) or (C,T), get {x.shape}")


def noise_injection_augmentation(data, noise_level=0.01):  
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

def random_global_scaling(data, scale_range=(0.9, 1.1)):
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return data * scale


def channel_wise_scaling(data, scale_range=(0.8, 1.2)):
    C = data.shape[0]
    scale = np.random.uniform(scale_range[0], scale_range[1], size=C)
    return data * scale[:, None]

def scale_one_channel_bands_energy(
        data,
        fs,
        target_channel=1,
        scale_range=(0.9, 1.1),
        k=0.3
):
    """
    仅对单个目标通道的 alpha/beta/theta 频带做自适应缩放。
    data: (C,T) numpy array
    """
    x = data.copy().astype(np.float32)
    C, T = x.shape

    # 目标通道
    ch = target_channel
    raw = x[ch]

    # 三个频带
    bands = {
        "theta": (4, 7),
        "alpha": (8, 12),
        "beta":  (13, 30),
    }

    # ===== 1. 各频带滤波 =====
    band_sigs = {}
    for name, (lo, hi) in bands.items():
        band_sigs[name] = bandpass_filter_np(raw, fs, lo, hi)

    # ===== 2. 残差 =====
    residual = raw.copy()
    for sig in band_sigs.values():
        residual -= sig

    # ===== 3. 能量 =====
    energies = np.array([(sig**2).mean() for sig in band_sigs.values()])

    # ===== 4. 自适应 alpha =====
    mu = energies.mean()
    sigma = energies.std() + 1e-8
    z = (energies - mu) / sigma

    alpha = 1.0 + k * z
    alpha = np.clip(alpha, scale_range[0], scale_range[1])

    # 对应回 dict
    alpha_dict = {name: alpha[i] for i, name in enumerate(bands.keys())}

    # ===== 5. 重新构建目标通道 =====
    new_ch = residual.copy()
    for name, sig in band_sigs.items():
        new_ch += alpha_dict[name] * sig

    x[ch] = new_ch.astype(np.float32)
    return x


def scale_all_channels_bands_energy(
        data,
        fs,
        scale_range=(0.9, 1.1),
        k=0.3
):
    """
    对所有通道的 theta/alpha/beta 频带做自适应能量缩放
    data: (C, T) numpy array
    return: (C, T) numpy array
    """
    x = data.copy().astype(np.float32)
    C, T = x.shape

    bands = {
        "theta": (4, 7),
        "alpha": (8, 12),
        "beta":  (13, 30),
    }

    new_x = np.zeros_like(x)

    # 对每个通道单独做 adaptive scaling
    for ch in range(C):
        raw = x[ch]  # (T,)

        # ===== 1. 三个频带滤波 =====
        band_sigs = {}
        for name, (lo, hi) in bands.items():
            band_sigs[name] = bandpass_filter_np(raw, fs, lo, hi)  # (T,)

        # ===== 2. 残差 =====
        residual = raw.copy()
        for sig in band_sigs.values():
            residual -= sig

        # ===== 3. 能量 =====
        energies = np.array([(sig**2).mean() for sig in band_sigs.values()])

        # ===== 4. 自适应 α =====
        mu = energies.mean()
        sigma = energies.std() + 1e-8
        z = (energies - mu) / sigma

        alpha = 1.0 + k * z
        alpha = np.clip(alpha, scale_range[0], scale_range[1])

        # ===== 5. 重构 =====
        new_ch = residual.copy()
        for idx, (name, _) in enumerate(bands.items()):
            new_ch += alpha[idx] * band_sigs[name]

        new_x[ch] = new_ch

    return new_x.astype(np.float32)
