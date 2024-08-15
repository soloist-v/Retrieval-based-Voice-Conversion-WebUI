import sys
import numpy as np
import soundfile
import torch

from infer.lib.audio import load_audio

if __name__ == "__main__":
    from configs.config import Config
    from infer.lib import rtrvc

    raw_path = "assets/voice/test.wav"
    out_path = "assets/output/test1.wav"
    sampling_rate = 48000
    target_sr = 16000
    f0method = "rmvpe"
    hop_size = 512

    pitch = 16
    formant = 0.33
    pth_path = "D:/Workspace/Python/Retrieval-based-Voice-Conversion-WebUI/assets/infer_model/Maaident1.pth"
    index_path = "logs/kikiV1.index"
    index_rate = 0.0
    n_cpu = 8.0
    # 为了多进程计算 harvest 发送计算请求
    inp_q = None  # multiprocessing.queues.Queue()
    # 为了多进程计算 harvest 接受计算结果
    opt_q = None  # multiprocessing.queues.Queue()
    # -------------Config----------------
    config = Config()
    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    # -------------Config----------------
    rvc = None
    model = rtrvc.RVC(
        pitch,
        formant,
        pth_path,
        index_path,
        index_rate,
        n_cpu,
        inp_q,
        opt_q,
        config,
        rvc,
    )
    # wav16k, sr = librosa.load(raw_path, sr=target_sr)
    wav16k = load_audio(raw_path, 16000)
    audio_max = np.abs(wav16k).max() / 0.95

    if audio_max > 1:
        wav16k /= audio_max
    block_frame_16k = 5600
    skip_head = 0
    return_length = len(wav16k) // 160
    x = torch.from_numpy(wav16k).to(config.device)
    with torch.no_grad():
        pred = model.infer(x, block_frame_16k, skip_head, return_length, f0method)
    audio = pred.cpu().numpy()
    # -------------------------------------------------

    # -------------------------------------------------
    soundfile.write(out_path, audio, sampling_rate)
