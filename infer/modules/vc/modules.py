import logging
import torch
import numpy as np
from typing import Optional
from fairseq import checkpoint_utils
from infer.lib.audio import load_audio
from infer.lib.infer_pack.models import SynthesizerTrnMs768NSFsid
from infer.modules.vc.pipeline import Pipeline

logger = logging.getLogger(__name__)


def load_hubert(model_path: str, device, is_half):
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task([model_path])
    hubert_model = models[0]
    hubert_model = hubert_model.to(device)
    if is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    return hubert_model.eval()


class VC:
    def __init__(self,
                 config,
                 rmvpe_path: str,
                 model_path: str,
                 hubert_path: str,
                 to_return_protect0,
                 to_return_protect1,
                 ):
        self.rmvpe_weights = rmvpe_path
        self.n_spk = None
        self.tgt_sr = None
        self.net_g = None
        self.pipeline: Optional[Pipeline] = None
        self.hubert_model = load_hubert(hubert_path, config.device, config.is_half)
        self.config = config
        self.to_return_protect0 = to_return_protect0
        self.to_return_protect1 = to_return_protect1
        self.cpt = torch.load(model_path, map_location="cpu")
        self.tgt_sr = self.cpt["config"][-1]
        print("target_sr:>>", self.tgt_sr)
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
        self.net_g = SynthesizerTrnMs768NSFsid(*self.cpt["config"], is_half=self.config.is_half)
        del self.net_g.enc_q
        self.net_g.load_state_dict(self.cpt["weight"], strict=False)
        self.net_g.eval().to(self.config.device)
        if self.config.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()
        self.pipeline = Pipeline(self.tgt_sr, self.rmvpe_weights, self.config)

    def vc_single(self,
                  sid,
                  input_audio_path,
                  f0_up_key,
                  f0_method,
                  index_rate,
                  resample_sr,
                  rms_mix_rate,
                  protect,
                  ):
        assert input_audio_path, f"file not found: {input_audio_path}"
        f0_up_key = int(f0_up_key)
        audio = load_audio(input_audio_path, 16000)

        # 不是必须的
        # audio_max = np.abs(audio).max() / 0.95
        # if audio_max > 1:
        #     audio /= audio_max

        audio_opt = self.pipeline.pipeline(
            sid,
            self.hubert_model,
            self.net_g,
            audio,
            f0_up_key,
            f0_method,
            index_rate,
            self.tgt_sr,
            resample_sr,
            rms_mix_rate,
            protect,
        )
        if self.tgt_sr != resample_sr >= 16000:
            tgt_sr = resample_sr
        else:
            tgt_sr = self.tgt_sr
        return tgt_sr, audio_opt
