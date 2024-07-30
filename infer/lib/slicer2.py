import numpy as np


# 这个函数来自librosa库,用于计算音频信号的均方根(RMS)值
def get_rms(
        y,  # 输入音频信号
        frame_length=2048,  # 每帧的长度,默认2048样本
        hop_length=512,  # 帧之间的重叠长度,默认512样本
        pad_mode="constant",  # 填充模式,默认为常数填充
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
            sample_rate: int,  # 采样率
            threshold: float = -40.0,  # 音量阈值，单位为分贝
            min_length: int = 5000,  # 最小音频片段长度，单位为毫秒
            min_interval: int = 300,  # 最小间隔，单位为毫秒
            hop_size: int = 20,  # 跳跃大小，单位为毫秒
            max_sil_kept: int = 5000,  # 保留的最大静音长度，单位为毫秒
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
                   :, begin * self.hop_size: min(waveform.shape[1], end * self.hop_size)
                   ]
        else:
            # 如果是一维数组（单声道），直接切片
            # 开始位置：begin * self.hop_size
            # 结束位置：min(waveform.shape[0], end * self.hop_size)，防止越界
            return waveform[
                   begin * self.hop_size: min(waveform.shape[0], end * self.hop_size)
                   ]

    # 定义一个名为slice的方法,用于将音频波形切片
    def slice(self, waveform):
        # 如果输入是多通道音频，将其转换为单通道；否则保持原样
        samples = waveform.mean(axis=0) if len(waveform.shape) > 1 else waveform

        # 如果音频长度小于等于最小长度阈值，直接返回整个音频
        if samples.shape[0] <= self.min_length:
            return [waveform]

        # 计算音频的RMS（均方根）值，用于判断音量大小
        rms_list = get_rms(
            y=samples, frame_length=self.win_size, hop_length=self.hop_size
        ).squeeze(0)

        sil_tags = []  # 存储检测到的静音区间
        silence_start = None  # 当前静音区间的开始位置
        clip_start = 0  # 当前音频片段的开始位置

        # 遍历每一帧的RMS值
        for i, rms in enumerate(rms_list):
            # 如果当前帧RMS值小于阈值，认为是静音
            if rms < self.threshold:
                # 如果是新的静音开始，记录其位置
                silence_start = i if silence_start is None else silence_start
            elif silence_start is not None:
                # 如果当前帧不是静音，且之前检测到了静音开始，判断是否需要切片

                # 判断是否为开头的长静音
                is_leading_silence = silence_start == 0 and i > self.max_sil_kept
                # 判断是否为中间需要切片的静音
                need_slice_middle = (
                        i - silence_start >= self.min_interval
                        and i - clip_start >= self.min_length
                )

                # 如果需要切片
                if is_leading_silence or need_slice_middle:
                    silence_length = i - silence_start  # 计算静音长度

                    # 处理短静音
                    if silence_length <= self.max_sil_kept:
                        # 在静音区间找到RMS最小的位置作为切点
                        pos = silence_start + np.argmin(rms_list[silence_start:i+1])
                        sil_tag = (0, pos) if silence_start == 0 else (pos, pos)
                        clip_start = pos
                    # 处理中等长度静音
                    elif silence_length <= self.max_sil_kept * 2:
                        # 在静音中间区域寻找最佳切点
                        pos = i - self.max_sil_kept + np.argmin(rms_list[i-self.max_sil_kept:silence_start+self.max_sil_kept+1])
                        # 在静音开始和结束区域各找一个切点
                        pos_l = silence_start + np.argmin(rms_list[silence_start:silence_start+self.max_sil_kept+1])
                        pos_r = i - self.max_sil_kept + np.argmin(rms_list[i-self.max_sil_kept:i+1])
                        if silence_start == 0:
                            sil_tag = (0, pos_r)
                            clip_start = pos_r
                        else:
                            sil_tag = (min(pos_l, pos), max(pos_r, pos))
                            clip_start = max(pos_r, pos)
                    # 处理长静音
                    else:
                        # 在静音的开始和结束各取一个切点
                        pos_l = silence_start + np.argmin(rms_list[silence_start:silence_start+self.max_sil_kept+1])
                        pos_r = i - self.max_sil_kept + np.argmin(rms_list[i-self.max_sil_kept:i+1])
                        sil_tag = (0, pos_r) if silence_start == 0 else (pos_l, pos_r)
                        clip_start = pos_r

                    # 将找到的静音标签添加到列表中
                    sil_tags.append(sil_tag)

                # 重置静音开始位置
                silence_start = None

        # 处理音频尾部的静音
        total_frames = rms_list.shape[0]
        if silence_start is not None and total_frames - silence_start >= self.min_interval:
            silence_end = min(total_frames, silence_start + self.max_sil_kept)
            pos = silence_start + np.argmin(rms_list[silence_start:silence_end+1])
            sil_tags.append((pos, total_frames + 1))

        # 如果没有检测到需要切片的静音，返回原始音频
        if not sil_tags:
            return [waveform]

        # 根据检测到的静音标签切分音频
        chunks = []
        # 处理第一个非静音片段（如果存在）
        if sil_tags[0][0] > 0:
            chunks.append(self._apply_slice(waveform, 0, sil_tags[0][0]))

        # 处理中间的非静音片段
        for i in range(len(sil_tags) - 1):
            chunks.append(self._apply_slice(waveform, sil_tags[i][1], sil_tags[i + 1][0]))

        # 处理最后一个非静音片段（如果存在）
        if sil_tags[-1][1] < total_frames:
            chunks.append(self._apply_slice(waveform, sil_tags[-1][1], total_frames))

        # 返回切分后的音频片段列表
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
