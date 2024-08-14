import os
import sys
from multiprocessing import Manager as M
import fairseq
import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Resample

from configs.config import Config
from infer.lib.jit.get_synthesizer import get_synthesizer
from infer.lib.rmvpe import RMVPE

now_dir = os.getcwd()
sys.path.append(now_dir)

mm = M()


def printt(strr, *args):
    if len(args) == 0:
        print(strr)
    else:
        print(strr % args)


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
        初始化
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
            printt("Index search enabled")
        self.pth_path: str = pth_path
        self.index_path = index_path
        self.index_rate = index_rate
        self.cache_pitch: torch.Tensor = torch.zeros(
            1024, device=self.device, dtype=torch.long
        )
        self.cache_pitchf = torch.zeros(
            1024, device=self.device, dtype=torch.float32
        )

        self.resample_kernel = {}

        printt("Loading rmvpe model")
        self.model_rmvpe = RMVPE(rmvpe_path, is_half=self.is_half, device=self.device)

        models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([hubert], suffix="")
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
        self.version = cpt.get("version", "v1")
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
            printt("Index search enabled")
        self.index_rate = new_index_rate

    def get_f0_post(self, f0):
        if not torch.is_tensor(f0):
            f0 = torch.from_numpy(f0)
        f0 = f0.float().to(self.device).squeeze()
        f0_mel = 1127 * torch.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * 254 / (self.f0_mel_max - self.f0_mel_min) + 1
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
        feats = (self.model.final_proj(logits[0]) if self.version == "v1" else logits[0])
        feats = torch.cat((feats, feats[:, -1:, :]), 1)

        p_len = input_wav.shape[0] // 160
        factor = pow(2, self.formant_shift / 12)
        return_length2 = int(np.ceil(return_length * factor))
        # 计算f0
        f0_extractor_frame = block_frame_16k + 800
        f0_extractor_frame = 5120 * ((f0_extractor_frame - 1) // 5120 + 1) - 160
        pitch, pitchf = self.get_f0_rmvpe(input_wav[-f0_extractor_frame:], self.f0_up_key - self.formant_shift)
        shift = block_frame_16k // 160
        self.cache_pitch[:-shift] = self.cache_pitch[shift:].clone()
        self.cache_pitchf[:-shift] = self.cache_pitchf[shift:].clone()
        self.cache_pitch[4 - pitch.shape[0]:] = pitch[3:-1]
        self.cache_pitchf[4 - pitch.shape[0]:] = pitchf[3:-1]
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
                self.resample_kernel[upp_res] = Resample(orig_freq=upp_res,
                                                         new_freq=self.tgt_sr // 100,
                                                         dtype=torch.float32).to(self.device)
            infered_audio = self.resample_kernel[upp_res](infered_audio[:, : return_length * upp_res])
        return infered_audio.squeeze()
