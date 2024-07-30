import os
import sys
import logging
import configs.config
from infer.lib.rmvpe import RMVPE
from functools import lru_cache
from time import time as ttime
import librosa
import numpy as np
import pyworld
import torch
import torch.nn.functional as F
from scipy import signal

logger = logging.getLogger(__name__)

now_dir = os.getcwd()
sys.path.append(now_dir)

input_audio_path2wav = {}


@lru_cache
def cache_harvest_f0(input_audio_path, fs, f0max, f0min, frame_period):
    audio = input_audio_path2wav[input_audio_path]
    f0, t = pyworld.harvest(
        audio,
        fs=fs,
        f0_ceil=f0max,
        f0_floor=f0min,
        frame_period=frame_period,
    )
    f0 = pyworld.stonemask(audio, f0, t, fs)
    return f0


def change_rms(data1, sr1, data2, sr2, rate):  # 1是输入音频，2是输出音频,rate是2的占比
    rms1 = librosa.feature.rms(
        y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )  # 每半秒一个点
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(
        rms1.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(
        rms2.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 *= (
            torch.pow(rms1, torch.tensor(1 - rate))
            * torch.pow(rms2, torch.tensor(rate - 1))
    ).numpy()
    return data2


class Pipeline(object):
    def __init__(self, tgt_sr, rmvpe_weights: str, config: configs.config.Config):
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)
        # 使用Butterworth滤波器创建高通滤波器，N=5表示滤波器阶数，Wn=48表示截止频率，
        # btype="high"表示高通滤波器，fs=16000表示采样率为16000Hz。

        self.x_pad, self.x_query, self.x_center, self.x_max, self.is_half = (
            config.x_pad,  # 填充时长，单位为秒
            config.x_query,  # 查询时间窗口，单位为秒
            config.x_center,  # 查询中心点，单位为秒
            config.x_max,  # 最大免查询时长，单位为秒
            config.is_half,  # 是否使用半精度浮点数进行计算
        )
        self.rmvpe_weights = rmvpe_weights
        self.sr = 16000  # hubert输入采样率
        self.window = 160  # 每帧点数
        self.t_pad = self.sr * self.x_pad  # 每条前后pad时间
        self.t_pad_tgt = tgt_sr * self.x_pad  # 计算目标采样率下的pad时间
        self.t_pad2 = self.t_pad * 2  # 计算总pad时间（前后pad时间之和）
        self.t_query = self.sr * self.x_query  # 查询切点前后查询时间，单位为采样点数
        self.t_center = self.sr * self.x_center  # 查询切点位置，单位为采样点数
        self.t_max = self.sr * self.x_max  # 免查询时长阈值，单位为采样点数
        self.device = config.device
        self.model_rmvpe = RMVPE(self.rmvpe_weights, is_half=self.is_half, device=self.device)

    def get_f0(self, x, f0_up_key, f0_method, inp_f0=None):
        global input_audio_path2wav
        assert f0_method == "rmvpe"

        # 设置f0的最小值和最大值
        f0_min = 50
        f0_max = 1100

        # 将f0的最小值和最大值转换为mel尺度
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)

        f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)  # 使用RMVPE模型从音频中推断出f0，thred=0.03为阈值

        f0 *= pow(2, f0_up_key / 12)  # 将推断出的f0根据f0_up_key进行音高调整，f0_up_key为音高提升的半音数

        tf0 = self.sr // self.window  # 计算每秒f0点数，self.sr为采样率，self.window为每帧点数

        if inp_f0 is not None:
            # 计算输入f0数据的时间跨度，并将其转换为相应的f0点数
            delta_t = np.round((inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1).astype("int16")

            # 使用插值法将输入的f0数据转换为目标长度的f0数据
            replace_f0 = np.interp(list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1])

            # 确定替换f0数据的长度，并将其插入到原始f0数据中
            shape = f0[self.x_pad * tf0: self.x_pad * tf0 + len(replace_f0)].shape[0]
            f0[self.x_pad * tf0: self.x_pad * tf0 + len(replace_f0)] = replace_f0[:shape]

        f0bak = f0.copy()  # 备份原始f0数据
        f0_mel = 1127 * np.log(1 + f0 / 700)  # 将f0数据转换为mel尺度

        # 对mel尺度的f0数据进行归一化处理，限制其范围在1到255之间
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255

        # 将处理后的f0数据取整并转换为整数类型
        f0_coarse = np.rint(f0_mel).astype(np.int32)

        # 返回处理后的f0数据和原始f0数据
        return f0_coarse, f0bak  # 1-0

    def vc(
            self,
            model,       # 模型对象，用于提取音频特征
            net_g,       # 神经网络对象，用于生成新的音频
            sid,         # 说话人ID，整数类型
            audio0,      # 原始音频数据，NumPy数组，范围为[-1, 1]
            pitch,       # 音高数据，PyTorch张量，浮点数，可选
            pitchf,      # 音高频率，PyTorch张量，浮点数，可选
            index,       # 特征索引，搜索相似特征的对象，可选
            big_npy,     # 大的特征数组，NumPy数组，可选
            index_rate,  # 索引特征的混合比率，浮点数，范围为[0, 1]
            protect,     # 保护系数，浮点数，范围为[0, 1]
    ):
        # 将输入音频转换为PyTorch张量
        feats = torch.from_numpy(audio0)

        # 判断是否使用半精度浮点数
        if self.is_half:
            feats = feats.half()
        else:
            feats = feats.float()

        # 如果是双声道音频，将其转为单声道
        if feats.dim() == 2:
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()

        # 调整张量形状为1xN
        feats = feats.view(1, -1)

        # 创建与feats相同形状的填充掩码，全为False
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)

        # 准备模型输入
        inputs = {
            "source": feats.to(self.device),  # 输入音频数据
            "padding_mask": padding_mask,     # 填充掩码
            "output_layer": 12,               # 提取第12层的特征
        }

        # 记录时间
        t0 = ttime()

        # 使用模型提取特征
        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats = logits[0]

        # 如果protect < 0.5 并且 pitch 和 pitchf 都不为空，则备份特征
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = feats.clone()

        # 如果index, big_npy均不为空且index_rate不为0，则进行特征调整
        # if (
        #         not isinstance(index, type(None))
        #         and not isinstance(big_npy, type(None))
        #         and index_rate != 0
        # ):
        #     npy = feats[0].cpu().numpy()
        #     if self.is_half:
        #         npy = npy.astype("float32")
        #
        #     # 使用索引搜索相似特征
        #     score, ix = index.search(npy, k=8)
        #     weight = np.square(1 / score)
        #     weight /= weight.sum(axis=1, keepdims=True)
        #     npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
        #
        #     if self.is_half:
        #         npy = npy.astype("float16")
        #     feats = (
        #             torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
        #             + (1 - index_rate) * feats
        #     )

        # 对特征进行插值操作，调整特征的时间尺度
        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        # 如果protect < 0.5 并且 pitch 和 pitchf 都不为空，则对备份的特征也进行相同的插值操作
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )

        # 记录时间
        t1 = ttime()

        # 计算每帧点数（p_len），用于音高调整
        p_len = audio0.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]

        # 根据protect参数调整特征
        if protect < 0.5 and pitch is not None and pitchf is not None:
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)

        # 将p_len转换为张量
        p_len = torch.tensor([p_len], device=self.device).long()

        # 使用神经网络生成新的音频
        with torch.no_grad():
            hasp = pitch is not None and pitchf is not None
            arg = (feats, p_len, pitch, pitchf, sid) if hasp else (feats, p_len, sid)
            audio1 = (net_g.infer(*arg)[0][0, 0]).data.cpu().float().numpy()
            del hasp, arg

        # 清理缓存
        del feats, p_len, padding_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # 记录时间
        t2 = ttime()

        # 返回生成的音频
        return audio1

    def pipeline(self,
                 sid,           # 说话人ID，整数，范围取决于训练集中的说话人数
                 model,         # 用于提取特征的模型对象
                 net_g,         # 用于生成音频的神经网络对象
                 audio,         # 输入音频，NumPy数组，32位浮点数，范围是-1到1
                 f0_up_key,     # 音高提升的半音数，整数
                 f0_method,     # 音高检测方法，字符串
                 index_rate,    # 索引特征的混合比率，浮点数，范围是0到1
                 tgt_sr,        # 目标采样率，整数
                 resample_sr,   # 重采样后的采样率，整数
                 rms_mix_rate,  # RMS混合比率，浮点数，范围是0到1
                 protect,       # 保护系数，浮点数，范围是0到1
                 ):
        index = big_npy = None

        # 对音频应用高通滤波器，系数为self.bh和self.ah，避免相位失真
        audio = signal.filtfilt(self.bh, self.ah, audio)

        # 在音频两端填充窗口大小的一半，用反射模式填充
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []

        # 如果填充后的音频长度超过最大长度阈值，则进行时间优化处理
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)

            # 计算音频信号的绝对值和，用于找到音频中能量最低的点
            for i in range(self.window):
                audio_sum += np.abs(audio_pad[i: i - self.window])

            # 在每个查询中心点处找到能量最低的点
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t - self.t_query
                    + np.where(
                        audio_sum[t - self.t_query: t + self.t_query]
                        == audio_sum[t - self.t_query: t + self.t_query].min()
                    )[0][0]
                )

        s = 0
        audio_opt = []
        t = None
        t1 = ttime()

        # 在音频两端填充t_pad，用反射模式填充
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        inp_f0 = None

        # 将说话人ID转换为PyTorch张量
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()

        # 获取音高和音高频率
        pitch, pitchf = self.get_f0(audio_pad, f0_up_key, f0_method, inp_f0)
        pitch = pitch[:p_len]
        pitchf = pitchf[:p_len]

        # 根据设备类型调整音高频率数据类型
        if "mps" not in str(self.device) or "xpu" not in str(self.device):
            pitchf = pitchf.astype(np.float32)

        # 将音高和音高频率转换为PyTorch张量
        pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
        pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()

        t2 = ttime()

        # 对优化时间点进行处理
        for t in opt_ts:  # opt_ts是优化后的时间点列表
            t = t // self.window * self.window  # 将时间点对齐到窗口大小的整数倍

            # 将经过处理的音频段添加到音频结果列表中
            audio_opt.append(
                self.vc(
                    model,
                    net_g,
                    sid,
                    audio_pad[s: t + self.t_pad2 + self.window],  # 从s到t + t_pad2 + 窗口大小的音频段
                    pitch[:, s // self.window: (t + self.t_pad2) // self.window],  # 对应的音高数据段
                    pitchf[:, s // self.window: (t + self.t_pad2) // self.window],  # 对应的音高频率数据段
                    index,
                    big_npy,
                    index_rate,
                    protect,
                )[self.t_pad_tgt: -self.t_pad_tgt]  # 去掉填充的部分
            )
            s = t  # 更新s为当前的t

        # 处理剩余音频段
        audio_opt.append(
            self.vc(
                model,
                net_g,
                sid,
                audio_pad[t:],  # 从最后一个t到结尾的音频段
                pitch[:, t // self.window:] if t is not None else pitch,  # 对应的音高数据段
                pitchf[:, t // self.window:] if t is not None else pitchf,  # 对应的音高频率数据段
                index,
                big_npy,
                index_rate,
                protect,
            )[self.t_pad_tgt: -self.t_pad_tgt]  # 去掉填充的部分
        )

        # 合并所有处理后的音频段
        audio_opt = np.concatenate(audio_opt)

        # 如果RMS混合比率不为1，则进行RMS混合处理
        if rms_mix_rate != 1:
            audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, rms_mix_rate)

        # 如果目标采样率和重采样率不同，则进行重采样
        if tgt_sr != resample_sr >= 16000:
            audio_opt = librosa.resample(
                audio_opt, orig_sr=tgt_sr, target_sr=resample_sr
            )

        # 归一化音频以防止溢出，并转换为16位整数
        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)

        # 删除不再需要的变量以释放内存
        del pitch, pitchf, sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        # 返回处理后的音频
        return audio_opt

