import multiprocessing
import os
import sys

from scipy import signal

now_dir = os.getcwd()
sys.path.append(now_dir)
print(*sys.argv[1:])
inp_root = sys.argv[1]
sample_rate = int(sys.argv[2])
number_of_process = int(sys.argv[3])
exp_dir = sys.argv[4]
noparallel = sys.argv[5] == "True"
per = float(sys.argv[6])
import os
import traceback

import librosa
import numpy as np
from scipy.io import wavfile

from infer.lib.audio import load_audio
from infer.lib.slicer2 import Slicer

f = open("%s/preprocess.log" % exp_dir, "a+")


def println(strr):
    print(strr)
    f.write("%s\n" % strr)
    f.flush()


class PreProcess:
    def __init__(self, sample_rate, exp_dir, per=3.7):
        self.slicer = Slicer(
            sample_rate=sample_rate,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )
        self.sample_rate = sample_rate
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=self.sample_rate)
        self.per = per
        self.overlap = 0.3
        self.tail = self.per + self.overlap
        self.max = 0.9
        self.alpha = 0.75
        self.exp_dir = exp_dir
        self.gt_wavs_dir = "%s/0_gt_wavs" % exp_dir
        self.wavs16k_dir = "%s/1_16k_wavs" % exp_dir
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)

    def norm_write(self, tmp_audio, idx0, idx1):
        # 计算音频数据的绝对值的最大值
        tmp_max = np.abs(tmp_audio).max()

        # 如果最大值超过2.5,认为音频可能有问题,打印信息并返回
        if tmp_max > 2.5:
            print("%s-%s-%s-filtered" % (idx0, idx1, tmp_max))
            return

        # 对音频进行归一化处理
        # 先将音频缩放到[-self.max*self.alpha, self.max*self.alpha]范围
        # 然后与原音频按self.alpha的比例混合,这样可以保留一些原始特征
        tmp_audio = (tmp_audio / tmp_max * (self.max * self.alpha)) + (
                1 - self.alpha
        ) * tmp_audio

        # 将处理后的音频保存为原采样率的wav文件
        wavfile.write(
            "%s/%s_%s.wav" % (self.gt_wavs_dir, idx0, idx1),  # 文件名格式为"idx0_idx1.wav"
            self.sample_rate,  # 使用原采样率
            tmp_audio.astype(np.float32),  # 将音频数据转换为32位浮点数
        )

        # 将音频重采样到16kHz
        tmp_audio = librosa.resample(
            tmp_audio, orig_sr=self.sample_rate, target_sr=16000
        )  # , res_type="soxr_vhq"  # 这是一个被注释掉的参数,可能用于指定重采样方法

        # 将重采样后的音频保存为16kHz的wav文件
        wavfile.write(
            "%s/%s_%s.wav" % (self.wavs16k_dir, idx0, idx1),  # 文件名格式与上面相同
            16000,  # 采样率为16kHz
            tmp_audio.astype(np.float32),  # 将音频数据转换为32位浮点数
        )

    def pipeline(self, path, idx0):
        try:
            # 加载音频文件,将其转换为指定采样率的数组, 这一步会导致采样率被指定
            audio = load_audio(path, self.sample_rate)

            # 对音频进行高通滤波,去除低频噪声
            audio = signal.lfilter(self.bh, self.ah, audio)

            # 使用slicer对象将音频切片,并遍历每个切片
            for slice_idx, audio_slice in enumerate(self.slicer.slice(audio)):
                # 初始化起始位置
                start = 0

                # 循环处理当前音频切片
                while start < len(audio_slice):
                    # 计算结束位置,每次处理self.per秒的音频
                    end = start + int(self.per * self.sample_rate)

                    # 提取当前处理的音频片段
                    tmp_audio = audio_slice[start:end]

                    # 如果剩余音频长度小于self.tail秒,说明是最后一个片段
                    if len(tmp_audio) < self.tail * self.sample_rate:
                        # 处理最后一个片段,取剩余所有音频
                        tmp_audio = audio_slice[start:]
                        # 对最后一个片段进行规范化处理并写入文件
                        self.norm_write(tmp_audio, idx0, slice_idx + 1)
                        # 结束当前切片的处理
                        break

                    # 对非最后片段进行规范化处理并写入文件
                    self.norm_write(tmp_audio, idx0, slice_idx + 1)

                    # 更新下一个片段的起始位置,考虑重叠部分
                    start += int(self.sample_rate * (self.per - self.overlap))

            print(f"{path}\t-> Success")
        except Exception as e:
            print(f"{path}\t-> {traceback.format_exc()}")

    def pipeline_mp(self, infos):
        for path, idx0 in infos:
            self.pipeline(path, idx0)

    def pipeline_mp_inp_dir(self, inp_root, n_p):
        try:
            infos = [
                ("%s/%s" % (inp_root, name), idx)
                for idx, name in enumerate(sorted(list(os.listdir(inp_root))))
            ]
            if noparallel:
                for i in range(n_p):
                    self.pipeline_mp(infos[i::n_p])
            else:
                ps = []
                for i in range(n_p):
                    p = multiprocessing.Process(
                        target=self.pipeline_mp, args=(infos[i::n_p],)
                    )
                    ps.append(p)
                    p.start()
                for i in range(n_p):
                    ps[i].join()
        except:
            println("Fail. %s" % traceback.format_exc())


def preprocess_trainset(inp_root, sr, n_p, exp_dir, per):
    pp = PreProcess(sr, exp_dir, per)
    println("start preprocess")
    pp.pipeline_mp_inp_dir(inp_root, n_p)
    println("end preprocess")


if __name__ == "__main__":
    preprocess_trainset(inp_root, sample_rate, number_of_process, exp_dir, per)
