import os

from infer.modules.onnx.export import export_onnx

os.environ["export_onnx"] = "True"
export_onnx(r"assets\infer_model\Maaident1.pth", "test.onnx")
