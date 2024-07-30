import sys
import soundfile
import torch

from os import listdir, path, mkdir
from configs.config import Config
from infer.modules.vc.modules import VC


raw_path = r"F:/Datasets/RVC/Opencpop/wavs"
out_path = r"assets/output"
if not path.exists(out_path):
    mkdir(out_path)

rmvpe_path = "assets/rmvpe/rmvpe.pt"
model_path = "assets/weights/kikiv2.pth"
file_index = "assets/indices/kiki.index"
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

f0_up_key = 0
f0_file = None
f0_method = "rmvpe"

index_rate = 0
filter_radius = 0
resample_sr = 48000
rms_mix_rate = 0.25
protect = 0.33

if __name__ == '__main__':
    vc = VC(config, rmvpe_path, model_path, hubert_path, 0.33, 0.33)
    for filename in listdir(raw_path):
        input_path = f"{raw_path}/{filename}"
        output_path = f"{out_path}/{filename}"
        print(f"正在处理{input_path}...")
        (target_sr, sound) = vc.vc_single(
            sid,
            input_path,
            f0_up_key,
            f0_method,
            index_rate,
            resample_sr,
            rms_mix_rate,
            protect,
        )
        soundfile.write(output_path, sound, target_sr)
        print(f"{output_path}处理完成！")
