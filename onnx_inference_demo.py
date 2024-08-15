import onnxruntime
import soundfile
import numpy as np


def infer_onnx():
    from infer.lib.infer_pack.onnx_inference import OnnxRVC

    hop_size = 512
    sampling_rate = 40000  # 采样率
    f0_up_key = 0  # 升降调
    sid = 0  # 角色ID
    f0_method = "dio"  # F0提取算法 harvest pm
    model_path = "test.onnx"  # 模型的完整路径
    vec_path = "assets/vec/vec-768-layer-12.onnx"  # 内部自动补齐为 f"pretrained/{vec_name}.onnx" 需要onnx的vec模型
    wav_path = "assets/voice/test.wav"  # 输入路径或ByteIO实例
    out_path = "assets/output/test.wav"  # 输出路径或ByteIO实例
    model = OnnxRVC(
        model_path, vec_path=vec_path, sr=sampling_rate, hop_size=hop_size, device="cpu"
    )
    audio: np.ndarray = model.inference(
        wav_path, sid, f0_method=f0_method, f0_up_key=f0_up_key
    )
    print("audio dtype", audio.dtype)
    soundfile.write(out_path, audio, sampling_rate)


def infer_test():
    model = onnxruntime.InferenceSession("test.onnx")
    n = 239
    vec_channels = 768
    test_phone = np.random.rand(1, n, vec_channels).astype("float32")  # hidden unit
    test_phone_lengths = np.array([n], dtype="int64")  # hidden unit 长度（貌似没啥用）
    test_pitch = np.random.randint(
        size=(1, n), low=5, high=255, dtype="int64"
    )  # 基频（单位赫兹）
    test_pitchf = np.random.rand(1, n).astype("float32")  # nsf基频
    test_ds = np.array([0], dtype="int64")  # 说话人ID
    test_rnd = np.random.rand(1, 192, n).astype("float32")  # 噪声（加入随机因子）
    print("phone:", test_phone.shape)
    print("phone_lengths:", test_phone_lengths.shape, test_phone_lengths)
    print("pitch:", test_pitch.shape)
    print("pitchf:", test_pitchf.shape)
    print("ds:", test_ds.shape, test_ds)
    print("rnd:", test_rnd.shape)
    onnx_input = {
        model.get_inputs()[0].name: test_phone,
        model.get_inputs()[1].name: test_phone_lengths,
        model.get_inputs()[2].name: test_pitch,
        model.get_inputs()[3].name: test_pitchf,
        model.get_inputs()[4].name: test_ds,
        model.get_inputs()[5].name: test_rnd,
    }
    res = model.run(None, onnx_input)
    print(res[0].shape)


if __name__ == "__main__":
    infer_onnx()
