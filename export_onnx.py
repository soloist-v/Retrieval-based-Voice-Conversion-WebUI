import yaml
import torch

from infer.lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM

if __name__ == "__main__":
    model_path = r"assets\infer_model\test014.pth"
    save_path = r"test.onnx"
    cfg = yaml.full_load(open("configs/v2/48k.json"))
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    vec_channels = 768
    device = "cpu"
    sr = "48k"
    sr2sr = {
        "32k": 32000,
        "40k": 40000,
        "48k": 48000,
    }
    net_g = SynthesizerTrnMsNSFsidM(
        data_cfg["filter_length"] // 2 + 1,
        train_cfg["segment_size"] // data_cfg["hop_length"],
        **model_cfg,
        sr=sr,
        is_half=False,
        encoder_dim=vec_channels
    )  # fp32导出（C++要支持fp16必须手动将内存重新排列所以暂时不用fp16）
    weight = torch.load(model_path, map_location=device)
    net_g.load_state_dict(weight, strict=False)
    input_names = ["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"]
    output_names = ["audio"]
    # net_g.construct_spkmixmap() #多角色混合轨道导出
    n = 239
    test_phone = torch.rand(1, n, vec_channels).to(device)  # hidden unit
    test_phone_lengths = (
        torch.tensor([n]).long().to(device)
    )  # hidden unit 长度（貌似没啥用）
    test_pitch = torch.randint(size=(1, n), low=5, high=255).to(
        device
    )  # 基频（单位赫兹）
    test_pitchf = torch.rand(1, n).to(device)  # nsf基频
    test_ds = torch.LongTensor([0]).to(device)  # 说话人ID
    test_rnd = torch.rand(1, 192, n).to(device)  # 噪声（加入随机因子）
    torch.onnx.export(
        net_g,
        (
            test_phone,
            test_phone_lengths,
            test_pitch,
            test_pitchf,
            test_ds,
            test_rnd,
        ),
        save_path,
        dynamic_axes={
            "phone": [1],
            "pitch": [1],
            "pitchf": [1],
            "rnd": [2],
        },
        do_constant_folding=False,
        opset_version=17,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
    )
