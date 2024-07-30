import numpy as np


# 这个函数来自librosa库,用于计算音频信号的均方根(RMS)值
def get_rms(
        y,                      # 输入音频信号
        frame_length=2048,      # 每帧的长度,默认2048样本
        hop_length=512,         # 帧之间的重叠长度,默认512样本
        pad_mode="constant",    # 填充模式,默认为常数填充
):
    # 计算需要在信号两端添加的填充长度
    padding = (int(frame_length // 2), int(frame_length // 2))
    # 对输入信号进行填充
    y = np.pad(y, padding, mode=pad_mode)

    # 设置要操作的轴为最后一个轴
    axis = -1
    # 创建新的步长,用于帧切分
    out_strides = y.strides + tuple([y.strides[axis]])
    # 调整输出形状,减去帧长度-1以避免越界
    x_shape_trimmed = list(y.shape)
    x_shape_trimmed[axis] -= frame_length - 1
    out_shape = tuple(x_shape_trimmed) + tuple([frame_length])
    # 使用stride_tricks创建帧视图,而不复制数据
    xw = np.lib.stride_tricks.as_strided(y, shape=out_shape, strides=out_strides)
    # 调整帧轴的位置
    if axis < 0:
        target_axis = axis - 1
    else:
        target_axis = axis + 1
    xw = np.moveaxis(xw, -1, target_axis)
    # 沿目标轴进行下采样
    slices = [slice(None)] * xw.ndim
    slices[axis] = slice(0, None, hop_length)
    x = xw[tuple(slices)]

    # 计算功率(平方后取平均)
    power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)

    # 返回均方根值(功率的平方根)
    return np.sqrt(power)


class Slicer:
    def __init__(
            self,
            sample_rate: int,                    # 采样率
            threshold: float = -40.0,   # 音量阈值，单位为分贝
            min_length: int = 5000,     # 最小音频片段长度，单位为毫秒
            min_interval: int = 300,    # 最小间隔，单位为毫秒
            hop_size: int = 20,         # 跳跃大小，单位为毫秒
            max_sil_kept: int = 5000,   # 保留的最大静音长度，单位为毫秒
    ):
        # 检查参数是否满足条件：最小长度 >= 最小间隔 >= 跳跃大小
        if not min_length >= min_interval >= hop_size:
            raise ValueError(
                "The following condition must be satisfied: min_length >= min_interval >= hop_size"
            )
        # 检查参数是否满足条件：最大保留静音 >= 跳跃大小
        if not max_sil_kept >= hop_size:
            raise ValueError(
                "The following condition must be satisfied: max_sil_kept >= hop_size"
            )

        min_interval = sample_rate * min_interval / 1000  # 将最小间隔从毫秒转换为采样点数
        self.threshold = 10 ** (threshold / 20.0)  # 将阈值从分贝转换为振幅比例
        self.hop_size = round(sample_rate * hop_size / 1000)  # 将跳跃大小从毫秒转换为采样点数
        self.win_size = min(round(min_interval), 4 * self.hop_size)  # 设置窗口大小，取最小间隔和4倍跳跃大小中的较小值
        self.min_length = round(sample_rate * min_length / 1000 / self.hop_size)  # 将最小长度从毫秒转换为跳跃次数
        self.min_interval = round(min_interval / self.hop_size)  # 将最小间隔从采样点数转换为跳跃次数
        self.max_sil_kept = round(sample_rate * max_sil_kept / 1000 / self.hop_size)  # 将最大保留静音从毫秒转换为跳跃次数

    def _apply_slice(self, waveform, begin, end):
        # begin & end : 切片的起始/结束位置(以 hop_size 为单位的索引）
        # 检查波形是否为多维数组（例如立体声）
        if len(waveform.shape) > 1:
            # 如果是多维数组，在第二个维度上切片
            # 开始位置：begin * self.hop_size
            # 结束位置：min(waveform.shape[1], end * self.hop_size)，防止越界
            return waveform[
                   :, begin * self.hop_size : min(waveform.shape[1], end * self.hop_size)
                   ]
        else:
            # 如果是一维数组（单声道），直接切片
            # 开始位置：begin * self.hop_size
            # 结束位置：min(waveform.shape[0], end * self.hop_size)，防止越界
            return waveform[
                   begin * self.hop_size : min(waveform.shape[0], end * self.hop_size)
                   ]

    # @timeit
    def slice(self, waveform):
        # 如果波形是多维的（如立体声），取平均值转为单声道
        if len(waveform.shape) > 1:
            samples = waveform.mean(axis=0)
        else:
            samples = waveform

        # 如果样本长度小于等于最小长度，直接返回整个波形
        if samples.shape[0] <= self.min_length:
            return [waveform]

        # 计算均方根（RMS）值列表
        rms_list = get_rms(
            y=samples, frame_length=self.win_size, hop_length=self.hop_size
        ).squeeze(0)

        sil_tags = []  # 用于存储静音区间的标签
        silence_start = None  # 静音开始的位置
        clip_start = 0  # 当前音频片段的开始位置

        # 遍历RMS值列表
        for i, rms in enumerate(rms_list):
            # 如果当前帧的RMS值小于阈值，认为是静音
            if rms < self.threshold:
                # 记录静音开始的位置
                if silence_start is None:
                    silence_start = i
                continue

            # 如果不是静音且未记录静音开始，继续循环
            if silence_start is None:
                continue

            # 判断是否需要切片
            is_leading_silence = silence_start == 0 and i > self.max_sil_kept
            need_slice_middle = (
                    i - silence_start >= self.min_interval
                    and i - clip_start >= self.min_length
            )

            # 如果不需要切片，重置静音开始位置并继续循环
            if not is_leading_silence and not need_slice_middle:
                silence_start = None
                continue

            # 需要切片，记录要移除的静音帧范围
            if i - silence_start <= self.max_sil_kept:
                # 在静音区间找到RMS最小的位置
                pos = rms_list[silence_start : i + 1].argmin() + silence_start
                if silence_start == 0:
                    sil_tags.append((0, pos))
                else:
                    sil_tags.append((pos, pos))
                clip_start = pos
            elif i - silence_start <= self.max_sil_kept * 2:
                # 在较长的静音区间中找到最佳切割点
                pos = rms_list[
                      i - self.max_sil_kept : silence_start + self.max_sil_kept + 1
                      ].argmin()
                pos += i - self.max_sil_kept
                pos_l = (
                        rms_list[
                        silence_start : silence_start + self.max_sil_kept + 1
                        ].argmin()
                        + silence_start
                )
                pos_r = (
                        rms_list[i - self.max_sil_kept : i + 1].argmin()
                        + i
                        - self.max_sil_kept
                )
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                    clip_start = pos_r
                else:
                    sil_tags.append((min(pos_l, pos), max(pos_r, pos)))
                    clip_start = max(pos_r, pos)
            else:
                # 对于非常长的静音区间，在两端各取一段
                pos_l = (
                        rms_list[
                        silence_start : silence_start + self.max_sil_kept + 1
                        ].argmin()
                        + silence_start
                )
                pos_r = (
                        rms_list[i - self.max_sil_kept : i + 1].argmin()
                        + i
                        - self.max_sil_kept
                )
                if silence_start == 0:
                    sil_tags.append((0, pos_r))
                else:
                    sil_tags.append((pos_l, pos_r))
                clip_start = pos_r
            silence_start = None

        # 处理尾部静音
        total_frames = rms_list.shape[0]
        if (
                silence_start is not None
                and total_frames - silence_start >= self.min_interval
        ):
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = rms_list[silence_start : silence_end + 1].argmin() + silence_start
            sil_tags.append((pos, total_frames + 1))

        # 应用切片并返回结果
        if len(sil_tags) == 0:
            return [waveform]
        else:
            chunks = []
            if sil_tags[0][0] > 0:
                chunks.append(self._apply_slice(waveform, 0, sil_tags[0][0]))
            for i in range(len(sil_tags) - 1):
                chunks.append(
                    self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0])
                )
            if sil_tags[-1][1] < total_frames:
                chunks.append(
                    self._apply_slice(waveform, sil_tags[-1][1], total_frames)
                )
            return chunks


def main():
    import os.path
    from argparse import ArgumentParser

    import librosa
    import soundfile

    parser = ArgumentParser()
    parser.add_argument("audio", type=str, help="The audio to be sliced")
    parser.add_argument(
        "--out", type=str, help="Output directory of the sliced audio clips"
    )
    parser.add_argument(
        "--db_thresh",
        type=float,
        required=False,
        default=-40,
        help="The dB threshold for silence detection",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        required=False,
        default=5000,
        help="The minimum milliseconds required for each sliced audio clip",
    )
    parser.add_argument(
        "--min_interval",
        type=int,
        required=False,
        default=300,
        help="The minimum milliseconds for a silence part to be sliced",
    )
    parser.add_argument(
        "--hop_size",
        type=int,
        required=False,
        default=10,
        help="Frame length in milliseconds",
    )
    parser.add_argument(
        "--max_sil_kept",
        type=int,
        required=False,
        default=500,
        help="The maximum silence length kept around the sliced clip, presented in milliseconds",
    )
    args = parser.parse_args()
    out = args.out
    if out is None:
        out = os.path.dirname(os.path.abspath(args.audio))
    audio, sr = librosa.load(args.audio, sr=None, mono=False)
    slicer = Slicer(
        sample_rate=sr,
        threshold=args.db_thresh,
        min_length=args.min_length,
        min_interval=args.min_interval,
        hop_size=args.hop_size,
        max_sil_kept=args.max_sil_kept,
    )
    chunks = slicer.slice(audio)
    if not os.path.exists(out):
        os.makedirs(out)
    for i, chunk in enumerate(chunks):
        if len(chunk.shape) > 1:
            chunk = chunk.T
        soundfile.write(
            os.path.join(
                out,
                f"%s_%d.wav"
                % (os.path.basename(args.audio).rsplit(".", maxsplit=1)[0], i),
            ),
            chunk,
            sr,
        )


if __name__ == "__main__":
    main()
