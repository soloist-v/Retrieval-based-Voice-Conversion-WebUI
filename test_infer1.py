import sys
import soundfile
import torch
from configs.config import Config
from infer.modules.vc.modules import VC

if __name__ == '__main__':
    raw_path = "assets/voice/test.wav"
    out_path = "assets/output/test.wav"
    rmvpe_path = "assets/rmvpe/rmvpe.pt"
    model_path = "assets/infer_model/Maaident1.pth"
    hubert_path = "assets/hubert/hubert_base.pt"
    config = Config()
    config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.is_half = False
    config.use_jit = False
    config.n_cpu = 16
    config.gpu_mem = 6
    config.python_cmd = sys.executable
    config.listen_port = 7865
    config.iscolab = False
    config.noparallel = False
    config.noautoopen = False
    config.dml = False
    config.nocheck = False
    config.update = False
    config.preprocess_per = 3.0
    config.x_pad = 1
    config.x_query = 6
    config.x_center = 38
    config.x_max = 41
    config.instead = ""
    sid = 0
    vc = VC(config, rmvpe_path, model_path, hubert_path, 0.33, 0.33)
    input_audio_path = raw_path
    f0_up_key = 14
    f0_file = None
    f0_method = "rmvpe"
    file_index = ""
    file_index2 = ""
    index_rate = 0
    filter_radius = 0
    resample_sr = 48000
    rms_mix_rate = 0.25
    protect = 0.33
    (target_sr, sound) = vc.vc_single(
        sid,
        input_audio_path,
        f0_up_key,
        f0_method,
        index_rate,
        resample_sr,
        rms_mix_rate,
        protect,
    )
    soundfile.write(out_path, sound, target_sr)
