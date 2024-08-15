import fairseq
import faiss
import librosa
import numpy as np
import soundfile
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Resample
from dataclasses import dataclass
from configs.config import Config
from infer.lib.jit.get_synthesizer import get_synthesizer
from infer.lib.rmvpe import RMVPE
from tools.torchgate import TorchGate


def phase_vocoder(a, b, fade_out, fade_in):
    window = torch.sqrt(fade_out * fade_in)
    fa = torch.fft.rfft(a * window)
    fb = torch.fft.rfft(b * window)
    absab = torch.abs(fa) + torch.abs(fb)
    n = a.shape[0]
    if n % 2 == 0:
        absab[1:-1] *= 2
    else:
        absab[1:] *= 2
    phia = torch.angle(fa)
    phib = torch.angle(fb)
    deltaphase = phib - phia
    deltaphase = deltaphase - 2 * np.pi * torch.floor(deltaphase / 2 / np.pi + 0.5)
    w = 2 * np.pi * torch.arange(n // 2 + 1).to(a) + deltaphase
    t = torch.arange(n).unsqueeze(-1).to(a) / n
    result = (
        a * (fade_out**2)
        + b * (fade_in**2)
        + torch.sum(absab * torch.cos(w * t + phia), -1) * window / n
    )
    return result


class RVC:
    def __init__(
        self,
        key,
        formant,
        pth_path,
        config: Config,
        rmvpe_path: str = "assets/rmvpe/rmvpe.pt",
        hubert: str = "assets/hubert/hubert_base.pt",
        index_path: str = "",
        index_rate: int = 0,
    ) -> None:
        """
        :param key:
        :param formant:
        :param pth_path: 生成器模型路径
        :param config:
        :param rmvpe_path:
        :param hubert:
        :param index_path:
        :param index_rate:
        """
        self.config = config
        self.device = config.device
        self.f0_up_key = key
        self.formant_shift = formant
        self.f0_min = 50
        self.f0_max = 1100
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)
        self.use_jit = self.config.use_jit
        self.is_half = config.is_half
        if index_rate != 0:
            self.index = faiss.read_index(index_path)
            self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
        self.pth_path: str = pth_path
        self.index_path = index_path
        self.index_rate = index_rate
        self.cache_pitch: torch.Tensor = torch.zeros(
            1024, device=self.device, dtype=torch.long
        )
        self.cache_pitchf = torch.zeros(1024, device=self.device, dtype=torch.float32)
        self.resample_kernel = {}
        self.model_rmvpe = RMVPE(rmvpe_path, is_half=self.is_half, device=self.device)
        models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task(
            [hubert], suffix=""
        )
        hubert_model = models[0]
        hubert_model = hubert_model.to(self.device)
        if self.is_half:
            hubert_model = hubert_model.half()
        else:
            hubert_model = hubert_model.float()
        hubert_model.eval()
        self.model = hubert_model
        self.net_g: nn.Module
        self.net_g, cpt = get_synthesizer(self.pth_path, self.device)
        self.tgt_sr = cpt["config"][-1]
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
        self.if_f0 = cpt.get("f0", 1)
        self.version = cpt.get("version", "v2")
        if self.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()

    def change_key(self, new_key):
        self.f0_up_key = new_key

    def change_formant(self, new_formant):
        self.formant_shift = new_formant

    def change_index_rate(self, new_index_rate):
        if new_index_rate != 0 and self.index_rate == 0:
            self.index = faiss.read_index(self.index_path)
            self.big_npy = self.index.reconstruct_n(0, self.index.ntotal)
        self.index_rate = new_index_rate

    def get_f0_post(self, f0):
        if not torch.is_tensor(f0):
            f0 = torch.from_numpy(f0)
        f0 = f0.float().to(self.device).squeeze()
        f0_mel = 1127 * torch.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (
            self.f0_mel_max - self.f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = torch.round(f0_mel).long()
        return f0_coarse, f0

    def get_f0_rmvpe(self, x, f0_up_key):
        f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
        f0 *= pow(2, f0_up_key / 12)
        return self.get_f0_post(f0)

    def infer(
        self,
        input_wav: torch.Tensor,
        block_frame_16k,
        skip_head,
        return_length,
    ) -> np.ndarray:
        if self.config.is_half:
            feats = input_wav.half().view(1, -1)
        else:
            feats = input_wav.float().view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)
        inputs = {
            "source": feats,
            "padding_mask": padding_mask,
            "output_layer": 9 if self.version == "v1" else 12,
        }
        logits = self.model.extract_features(**inputs)
        feats = self.model.final_proj(logits[0]) if self.version == "v1" else logits[0]
        feats = torch.cat((feats, feats[:, -1:, :]), 1)

        p_len = input_wav.shape[0] // 160
        factor = pow(2, self.formant_shift / 12)
        return_length2 = int(np.ceil(return_length * factor))
        # 计算f0
        f0_extractor_frame = block_frame_16k + 800
        f0_extractor_frame = 5120 * ((f0_extractor_frame - 1) // 5120 + 1) - 160
        pitch, pitchf = self.get_f0_rmvpe(
            input_wav[-f0_extractor_frame:], self.f0_up_key - self.formant_shift
        )
        shift = block_frame_16k // 160
        self.cache_pitch[:-shift] = self.cache_pitch[shift:].clone()
        self.cache_pitchf[:-shift] = self.cache_pitchf[shift:].clone()
        self.cache_pitch[4 - pitch.shape[0] :] = pitch[3:-1]
        self.cache_pitchf[4 - pitch.shape[0] :] = pitchf[3:-1]
        cache_pitch = self.cache_pitch[None, -p_len:]
        cache_pitchf = self.cache_pitchf[None, -p_len:] * return_length2 / return_length

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        feats = feats[:, :p_len, :]
        p_len = torch.LongTensor([p_len]).to(self.device)
        sid = torch.LongTensor([0]).to(self.device)
        skip_head = torch.LongTensor([skip_head])
        return_length2 = torch.LongTensor([return_length2])
        return_length = torch.LongTensor([return_length])

        infered_audio, _, _ = self.net_g.infer(
            feats,
            p_len,
            cache_pitch,
            cache_pitchf,
            sid,
            skip_head,
            return_length,
            return_length2,
        )

        infered_audio = infered_audio.squeeze(1).float()
        upp_res = int(np.floor(factor * self.tgt_sr // 100))
        if upp_res != self.tgt_sr // 100:
            if upp_res not in self.resample_kernel:
                self.resample_kernel[upp_res] = Resample(
                    orig_freq=upp_res, new_freq=self.tgt_sr // 100, dtype=torch.float32
                ).to(self.device)
            infered_audio = self.resample_kernel[upp_res](
                infered_audio[:, : return_length * upp_res]
            )
        return infered_audio.squeeze()


@dataclass
class Params:
    device: str  # cuda:0
    pth_path: str  # 生成器模型路径
    threshold: int  # 响应阈值|range=(-60, 0)|data.get("threhold", -60)
    pitch: int  # 音调设置|range=(-16, 16) | data.get("pitch", 0)
    formant: (
        float  # 性别因子/声线粗细|range=(-2, 2) | data.get("formant", 0.0)|step=0.05
    )
    index_rate: float  # Index Rate|range=(0.0, 1.0)|data.get("index_rate", 0)
    rms_mix_rate: float  # 响度因子 | range=(0.0, 1.0) | resolution=0.01 | data.get("rms_mix_rate", 0)
    f0method: str  # 是否开启高音算法 | default="rmvpe"
    block_time: float  # 采样长度 | range=(0.02, 1.5) | resolution=0.01 | data.get("block_time", 0.25)
    crossfade_time: float  # 淡入淡出长度 | range=(0.01, 0.15) | step=0.01 | data.get("crossfade_length", 0.05)
    extra_time: float  # 额外推理时长 | range=(0.05, 5.00) | data.get("extra_time", 2.5) | resolution=0.01
    I_noise_reduce: bool  # 输入降噪
    O_noise_reduce: bool  # 输出降噪
    use_pv: bool  # 启用相位声码器 | default = False
    # samplerate: int  # 模型对应的输入数据的采样率 16000
    channels: int  # 设置对应的通道数，对于对通道处理的方式是对多个通道取平均值


def inference(indata: np.ndarray, params: Params, rvc: RVC):
    samplerate = rvc.tgt_sr
    zc = samplerate // 100
    block_frame = int(np.round(params.block_time * samplerate / zc)) * zc

    block_frame_16k = 160 * block_frame // zc
    crossfade_frame = int(np.round(params.crossfade_time * samplerate / zc)) * zc
    sola_buffer_frame = min(crossfade_frame, 4 * zc)
    sola_search_frame = zc
    extra_frame = int(np.round(params.extra_time * samplerate / zc)) * zc
    input_wav: torch.Tensor = torch.zeros(
        extra_frame + crossfade_frame + sola_search_frame + block_frame,
        device=params.device,
        dtype=torch.float32,
    )
    input_wav_denoise: torch.Tensor = input_wav.clone()
    input_wav_res: torch.Tensor = torch.zeros(
        160 * input_wav.shape[0] // zc, device=params.device, dtype=torch.float32
    )
    rms_buffer: np.ndarray = np.zeros(4 * zc, dtype="float32")
    sola_buffer: torch.Tensor = torch.zeros(
        sola_buffer_frame, device=params.device, dtype=torch.float32
    )
    nr_buffer: torch.Tensor = sola_buffer.clone()
    output_buffer: torch.Tensor = input_wav.clone()
    skip_head = extra_frame // zc
    return_length = (block_frame + sola_buffer_frame + sola_search_frame) // zc
    fade_in_window: torch.Tensor = (
        torch.sin(
            0.5
            * np.pi
            * torch.linspace(
                0.0,
                1.0,
                steps=sola_buffer_frame,
                device=params.device,
                dtype=torch.float32,
            )
        )
        ** 2
    )
    fade_out_window: torch.Tensor = 1 - fade_in_window
    resampler = Resample(orig_freq=samplerate, new_freq=16000, dtype=torch.float32).to(
        params.device
    )
    if rvc.tgt_sr != samplerate:
        resampler2 = Resample(
            orig_freq=rvc.tgt_sr, new_freq=samplerate, dtype=torch.float32
        ).to(params.device)
    else:
        resampler2 = None
    tg = TorchGate(sr=samplerate, n_fft=4 * zc, prop_decrease=0.9).to(params.device)
    # -----------------------------------------start voice conversion--------------------------------------------
    indata = librosa.to_mono(indata.T)
    if params.threshold > -60:
        indata = np.append(rms_buffer, indata)
        rms = librosa.feature.rms(y=indata, frame_length=4 * zc, hop_length=zc)[:, 2:]
        rms_buffer[:] = indata[-4 * zc :]
        indata = indata[2 * zc - zc // 2 :]
        db_threshold = librosa.amplitude_to_db(rms, ref=1.0)[0] < params.threshold
        for i in range(db_threshold.shape[0]):
            if db_threshold[i]:
                indata[i * zc : (i + 1) * zc] = 0
        indata = indata[zc // 2 :]
    input_wav[:-block_frame] = input_wav[block_frame:].clone()
    input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(params.device)
    input_wav_res[:-block_frame_16k] = input_wav_res[block_frame_16k:].clone()
    # input noise reduction and resampling
    if params.I_noise_reduce:
        input_wav_denoise[:-block_frame] = input_wav_denoise[block_frame:].clone()
        input_wav = input_wav[-sola_buffer_frame - block_frame :]
        input_wav = tg(input_wav.unsqueeze(0), input_wav.unsqueeze(0)).squeeze(0)
        input_wav[:sola_buffer_frame] *= fade_in_window
        input_wav[:sola_buffer_frame] += nr_buffer * fade_out_window
        input_wav_denoise[-block_frame:] = input_wav[:block_frame]
        nr_buffer[:] = input_wav[block_frame:]
        input_wav_res[-block_frame_16k - 160 :] = resampler(
            input_wav_denoise[-block_frame - 2 * zc :]
        )[160:]
    else:
        input_wav_res[-160 * (indata.shape[0] // zc + 1) :] = resampler(
            input_wav[-indata.shape[0] - 2 * zc :]
        )[160:]
    # infer
    infer_wav = rvc.infer(input_wav_res, block_frame_16k, skip_head, return_length)
    if resampler2 is not None:
        infer_wav = resampler2(infer_wav)
    # output noise reduction
    if params.O_noise_reduce:
        output_buffer[:-block_frame] = output_buffer[block_frame:].clone()
        output_buffer[-block_frame:] = infer_wav[-block_frame:]
        infer_wav = tg(infer_wav.unsqueeze(0), output_buffer.unsqueeze(0)).squeeze(0)
    # volume envelop mixing
    if params.rms_mix_rate < 1:
        if params.I_noise_reduce:
            input_wav = input_wav_denoise[extra_frame:]
        else:
            input_wav = input_wav[extra_frame:]
        rms1 = librosa.feature.rms(
            y=input_wav[: infer_wav.shape[0]].cpu().numpy(),
            frame_length=4 * zc,
            hop_length=zc,
        )
        rms1 = torch.from_numpy(rms1).to(params.device)
        rms1 = F.interpolate(
            rms1.unsqueeze(0),
            size=infer_wav.shape[0] + 1,
            mode="linear",
            align_corners=True,
        )[0, 0, :-1]
        rms2 = librosa.feature.rms(
            y=infer_wav[:].detach().cpu().numpy(),
            frame_length=4 * zc,
            hop_length=zc,
        )
        rms2 = torch.from_numpy(rms2).to(params.device)
        rms2 = F.interpolate(
            rms2.unsqueeze(0),
            size=infer_wav.shape[0] + 1,
            mode="linear",
            align_corners=True,
        )[0, 0, :-1]
        rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
        infer_wav *= torch.pow(rms1 / rms2, torch.tensor(1 - params.rms_mix_rate))
    # SOLA algorithm from https://github.com/yxlllc/DDSP-SVC
    conv_input = infer_wav[None, None, : sola_buffer_frame + sola_search_frame]
    cor_nom = F.conv1d(conv_input, sola_buffer[None, None, :])
    cor_den = torch.sqrt(
        F.conv1d(
            conv_input**2,
            torch.ones(1, 1, sola_buffer_frame, device=params.device),
        )
        + 1e-8
    )
    sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])
    infer_wav = infer_wav[sola_offset:]
    infer_wav[:sola_buffer_frame] = phase_vocoder(
        sola_buffer, infer_wav[:sola_buffer_frame], fade_out_window, fade_in_window
    )
    sola_buffer[:] = infer_wav[block_frame : block_frame + sola_buffer_frame]
    outdata = infer_wav[:block_frame].repeat(params.channels, 1).t().cpu().numpy()
    return outdata


if __name__ == "__main__":
    args = Params(
        device="cuda:0",
        pth_path="data/model/Maaident1.pth",
        threshold=-60,
        pitch=13,
        formant=0.0,
        index_rate=0.0,
        rms_mix_rate=1.0,
        f0method="rmvpe",
        block_time=0.15,
        crossfade_time=0.01,
        extra_time=2.0,
        I_noise_reduce=False,
        O_noise_reduce=False,
        use_pv=False,
        channels=1,
    )
    config = Config()
    with torch.no_grad():
        rvc = RVC(args.pitch, args.formant, args.pth_path, config)
        target_sr = rvc.tgt_sr

        zc = target_sr // 100
        block_frame = int(np.round(args.block_time * target_sr / zc)) * zc

        audio_data, src_sr = librosa.load("data/test.wav", sr=rvc.tgt_sr)
        res = np.zeros((0,), dtype=np.float32)
        for start in range(0, audio_data.shape[0], block_frame):
            start -= block_frame
            start = max(0, start)
            indata = audio_data[start : start + block_frame]
            print("indata", len(indata))
            if len(indata) != block_frame:
                break
            audio_out = inference(indata, args, rvc).squeeze()
            print("res", len(audio_out))
            res = np.concatenate((res, audio_out), axis=0)
        soundfile.write("data/output/test.wav", res, target_sr)
